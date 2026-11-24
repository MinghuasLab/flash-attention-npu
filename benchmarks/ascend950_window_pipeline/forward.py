import os
import struct
import sys
import torch
import torch_npu
from flash_attn_npu_4 import flash_attn_func, get_scheduler_metadata

torch.npu.set_device(int(os.getenv("FAG_TEST_DEVICE", "0")))
torch.manual_seed(int(os.getenv("FAG_TEST_SEED","0")))
b, h, s, d = 1, 64, 4096, 256
q, k, v, dy = [torch.rand(b,h,s,d).to(device="npu",dtype=torch.bfloat16) for _ in range(4)]
for x in (q,k,v):
    x.requires_grad_(True)
pse = torch.rand(b,h,1,s).to(device="npu",dtype=torch.bfloat16)
sink = (torch.linspace(-1,1,h,device="npu") if int(os.getenv("FAG_TEST_SEED","0")) else
        torch.zeros(h,device="npu",dtype=torch.float32)).requires_grad_()
mask = torch.ones(2048,2048,dtype=torch.bool).triu_(1).npu()
qb,kb,vb,db = [x.transpose(1,2) for x in (q,k,v,dy)]
os.environ["FAG950_EXPERIMENT_BAND_512"]="1"
def cann_forward():
    o,mx,sm,*_ = torch_npu.npu_fusion_attention_v2(
        q,k,v,h,"BNSD",pse=pse,sink=sink,scale=d**-0.5,keep_prob=1.0,
        atten_mask=mask,pre_tokens=511,next_tokens=0,sparse_mode=4,pse_type=1)
    return o,mx,sm

raw=get_scheduler_metadata(max_seqlen_q=s,max_seqlen_k=s,nheads=h,nheads_kv=h,headdim=d,headdim_v=d,
    num_splits=1,window_size_left=511,window_size_right=0,
    seqused_k=torch.full((b,),s,dtype=torch.int32,device=q.device),qkv_dtype=q.dtype)
data=bytearray(raw.cpu().tolist())
def forward(meta):
    return flash_attn_func(qb,kb,vb,pse=pse,learnable_sink=sink,softmax_scale=d**-0.5,
        window_size=(511,0),return_lse=True,num_splits=1,scheduler_metadata=meta)
def config(code):
    if code==-1: return None
    if code==0: return raw
    variant=bytearray(data)
    struct.pack_into("<I",variant,96,0xf9500000+code)
    return torch.tensor(list(variant),device=q.device,dtype=torch.uint8)

modes=[int(v) for v in sys.argv[1:] if v!="accuracy"]
if "accuracy" in sys.argv:
    ref=cann_forward()
    cann_out=ref[0].transpose(1,2)
    cann_lse=ref[1][...,0]+ref[2][...,0].log()
    baseline=forward(raw)
    for mode in modes:
        for name,new,old,want in zip(("out","LSE"),forward(config(mode)),baseline,(cann_out,cann_lse)):
            diff=(new.float()-want.float()).abs()
            bound=torch.maximum(torch.full_like(diff,0.005), 0.005*want.float().abs())
            print(mode,name,"changed",torch.count_nonzero(new!=old).item(), "cann fails",
                  (diff>bound).sum().item(), "cann max",diff.max().item(),flush=True)
            assert torch.equal(new,old)
else:
    for _ in range(9): cann_forward()
    torch.npu.synchronize()
    for mode in modes:
        meta=config(mode)
        for _ in range(9): forward(meta)
        torch.npu.synchronize()
    for _ in range(9): cann_forward()
    torch.npu.synchronize()
