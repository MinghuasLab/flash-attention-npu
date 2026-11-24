import os
import sys
import torch
import torch_npu
from flash_attn_npu_4 import flash_attn_func

torch.npu.set_device(int(os.getenv("FAG_TEST_DEVICE", "0")))
torch.manual_seed(int(sys.argv[1]))
b, h, s, d = 1, 64, 4096, 256
q, k, v, dy = [torch.rand(b,h,s,d).to(device="npu",dtype=torch.bfloat16) for _ in range(4)]
for x in (q,k,v):
    x.requires_grad_(True)
pse = torch.rand(b,h,1,s).to(device="npu",dtype=torch.bfloat16)
sink = torch.linspace(-1,1,h,device="npu").requires_grad_() if int(sys.argv[1]) else torch.zeros(h,device="npu",dtype=torch.float32,requires_grad=True)
mask = torch.ones(2048,2048,dtype=torch.bool).triu_(1).npu()
qb,kb,vb,db = [x.transpose(1,2) for x in (q,k,v,dy)]
os.environ["FAG950_EXPERIMENT_BAND_512"]="1"
def cann_forward():
    o,mx,sm,*_ = torch_npu.npu_fusion_attention_v2(
        q,k,v,h,"BNSD",pse=pse,sink=sink,scale=d**-0.5,keep_prob=1.0,
        atten_mask=mask,pre_tokens=511,next_tokens=0,sparse_mode=4,pse_type=1)
    return o,mx,sm

def v4_forward():
    return flash_attn_func(qb,kb,vb,pse=pse,learnable_sink=sink,softmax_scale=d**-0.5,
                          window_size=(511,0),return_lse=True,num_splits=1)

def cann():
    out,mx,sm=cann_forward()
    return (out,mx[...,0]+sm[...,0].log(),*torch.autograd.grad(out,(q,k,v,sink),dy))

def v4():
    out,lse=v4_forward()
    return (out.transpose(1,2),lse,*torch.autograd.grad(out,(q,k,v,sink),db))


from flash_attn_npu_3.flash_attn_npu_interface_950 import _flash_attn_backward
import flash_attn_npu_3_950, flash_attn_npu_4_950
print(flash_attn_npu_3_950.__file__, flash_attn_npu_4_950.__file__, flush=True)
fo, fl = v4_forward()
dq,dk,dv=(torch.empty(x.shape, device=x.device, dtype=x.dtype) for x in (qb,kb,vb))
dsink=torch.empty_like(sink)
def run(mode):
    os.environ["FAG950_EXPERIMENT_BAND_DRAIN"]=str(mode)
    _flash_attn_backward(db,qb,kb,vb,fo,fl,dq=dq,dk=dk,dv=dv,dsink=dsink,pse=pse,sink=sink,
                         softmax_scale=d**-0.5,window_size_left=511,window_size_right=0)
    return dq,dk,dv,dsink


ref=cann()[2:]
base=[x.clone().transpose(1,2) if x.ndim==4 else x.clone() for x in run(0)]
for mode in (int(v) for v in sys.argv[2:]):
    raw=run(mode)
    for name,actual,prior,want in zip(("dQ","dK","dV","dSink"),raw,base,ref):
        got=actual.transpose(1,2) if actual.ndim==4 else actual
        err=(got.float()-want.float()).abs()
        base_err=(prior.float()-want.float()).abs()
        limit=torch.maximum(torch.full_like(err,0.005),0.005*want.float().abs())
        change=(got.float()-prior.float()).abs()
        fail=(err>limit).sum().item()
        base_fail=(base_err>limit).sum().item()
        print(int(sys.argv[1]),mode,name, "cann max",err.max().item(),"cann rms",err.square().mean().sqrt().item(),
              "cann fails", fail, "prior fails",base_fail, "prior max diff",change.max().item(),
              "prior changed",torch.count_nonzero(change).item(), flush=True)
        assert torch.isfinite(got).all() and fail<=base_fail
