import os
import sys
import torch
import torch_npu
from flash_attn_npu_4 import flash_attn_func

torch.npu.set_device(int(os.getenv("FAG_TEST_DEVICE", "0")))
torch.manual_seed(0)
b, h, s, d = 1, 64, 4096, 256
q, k, v, dy = [torch.rand(b,h,s,d).to(device="npu",dtype=torch.bfloat16) for _ in range(4)]
for x in (q,k,v):
    x.requires_grad_(True)
pse = torch.rand(b,h,1,s).to(device="npu",dtype=torch.bfloat16)
sink = torch.zeros(h,device="npu",dtype=torch.float32,requires_grad=True)
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


for mode in ("cann",0,269551368,269551368,0,"cann"):
    for _ in range(9):
        if mode=="cann":
            co,*_=cann_forward()
            torch.autograd.grad(co,(q,k,v,sink),dy)
        else:
            run(mode)
    torch.npu.synchronize()
