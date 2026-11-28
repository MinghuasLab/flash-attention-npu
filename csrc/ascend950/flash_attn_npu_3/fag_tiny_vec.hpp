#ifndef FAG_TINY_VEC_HPP
#define FAG_TINY_VEC_HPP

#ifdef __DAV_VEC__
namespace FagTiny {
using namespace AscendC::MicroAPI;

template<class T>
__simd_callee__ inline void LoadRow(__ubuf__ T* x, RegTensor<float>& even,
                                   RegTensor<float>& odd, MaskReg& m)
{
    RegTensor<T> packed;
    LoadAlign(packed,x);
    constexpr static CastTrait evenCast{RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                  AscendC::RoundMode::UNKNOWN};
    constexpr static CastTrait oddCast{RegLayout::ONE, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                 AscendC::RoundMode::UNKNOWN};
    Cast<float,T,evenCast>(even,packed,m);
    Cast<float,T,oddCast>(odd,packed,m);
}

template<class T>
__simd_callee__ inline void StoreRow(__ubuf__ T* x, RegTensor<float>& even,
                                    RegTensor<float>& odd, MaskReg& m)
{
    RegTensor<T> lo,hi,packed;
    constexpr static CastTrait evenCast{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
                                  AscendC::RoundMode::CAST_ROUND};
    constexpr static CastTrait oddCast{RegLayout::ONE, SatMode::SAT, MaskMergeMode::ZEROING,
                                 AscendC::RoundMode::CAST_ROUND};
    Cast<T,float,evenCast>(lo,even,m);
    Cast<T,float,oddCast>(hi,odd,m);
    Or(reinterpret_cast<RegTensor<uint16_t>&>(packed),
       reinterpret_cast<RegTensor<uint16_t>&>(lo),reinterpret_cast<RegTensor<uint16_t>&>(hi),m);
    StoreAlign<T,StoreDist::DIST_NORM_B16>(x,packed,m);
}

__simd_callee__ inline void Dot(RegTensor<float>& z, RegTensor<float>& a,
                               RegTensor<float>& b, RegTensor<float>& c,
                               RegTensor<float>& d, MaskReg& m)
{
    RegTensor<float> x,y,sum;
    Mul(x,a,c,m);
    Mul(y,b,d,m);
    Add(x,x,y,m);
    ReduceSum(sum,x,m);
    Duplicate<float,HighLowPart::LOWEST>(z,sum,m);
}

template<class T>
__simd_vf__ inline void Backward(__ubuf__ T* q,__ubuf__ T* k,__ubuf__ T* v,
                                 __ubuf__ T* dy,__ubuf__ T* o,__ubuf__ float* lse,
                                 __ubuf__ T* dq,__ubuf__ T* dk,__ubuf__ T* dv,float scale)
{
    auto m=CreateMask<float,MaskPattern::ALL>();
    auto m16=CreateMask<T,MaskPattern::ALL>();
    RegTensor<float> q0e,q0o;
    LoadRow(q+0,q0e,q0o,m16);
    RegTensor<float> q1e,q1o;
    LoadRow(q+128,q1e,q1o,m16);
    RegTensor<float> q2e,q2o;
    LoadRow(q+256,q2e,q2o,m16);
    RegTensor<float> k0e,k0o;
    LoadRow(k+0,k0e,k0o,m16);
    RegTensor<float> k1e,k1o;
    LoadRow(k+128,k1e,k1o,m16);
    RegTensor<float> k2e,k2o;
    LoadRow(k+256,k2e,k2o,m16);
    RegTensor<float> v0e,v0o;
    LoadRow(v+0,v0e,v0o,m16);
    RegTensor<float> v1e,v1o;
    LoadRow(v+128,v1e,v1o,m16);
    RegTensor<float> v2e,v2o;
    LoadRow(v+256,v2e,v2o,m16);
    RegTensor<float> dy0e,dy0o;
    LoadRow(dy+0,dy0e,dy0o,m16);
    RegTensor<float> dy1e,dy1o;
    LoadRow(dy+128,dy1e,dy1o,m16);
    RegTensor<float> dy2e,dy2o;
    LoadRow(dy+256,dy2e,dy2o,m16);
    RegTensor<float> delta0;
    {
        RegTensor<float> oe,oo;
        LoadRow(o+0,oe,oo,m16);
        Dot(delta0,dy0e,dy0o,oe,oo,m);
    }
    RegTensor<float> p00, ds00;
    {
        RegTensor<float> l;
        LoadAlign<float,LoadDist::DIST_BRC_B32>(l,lse+0);
        Dot(p00,q0e,q0o,k0e,k0o,m);
        Muls(p00,p00,scale,m);
        Sub(p00,p00,l,m);
        Exp(p00,p00,m);
        Dot(ds00,dy0e,dy0o,v0e,v0o,m);
        Sub(ds00,ds00,delta0,m);
        Mul(ds00,ds00,p00,m);
        Muls(ds00,ds00,scale,m);
    }
    RegTensor<float> delta1;
    {
        RegTensor<float> oe,oo;
        LoadRow(o+128,oe,oo,m16);
        Dot(delta1,dy1e,dy1o,oe,oo,m);
    }
    RegTensor<float> p10, ds10;
    {
        RegTensor<float> l;
        LoadAlign<float,LoadDist::DIST_BRC_B32>(l,lse+1);
        Dot(p10,q1e,q1o,k0e,k0o,m);
        Muls(p10,p10,scale,m);
        Sub(p10,p10,l,m);
        Exp(p10,p10,m);
        Dot(ds10,dy1e,dy1o,v0e,v0o,m);
        Sub(ds10,ds10,delta1,m);
        Mul(ds10,ds10,p10,m);
        Muls(ds10,ds10,scale,m);
    }
    RegTensor<float> p11, ds11;
    {
        RegTensor<float> l;
        LoadAlign<float,LoadDist::DIST_BRC_B32>(l,lse+1);
        Dot(p11,q1e,q1o,k1e,k1o,m);
        Muls(p11,p11,scale,m);
        Sub(p11,p11,l,m);
        Exp(p11,p11,m);
        Dot(ds11,dy1e,dy1o,v1e,v1o,m);
        Sub(ds11,ds11,delta1,m);
        Mul(ds11,ds11,p11,m);
        Muls(ds11,ds11,scale,m);
    }
    RegTensor<float> delta2;
    {
        RegTensor<float> oe,oo;
        LoadRow(o+256,oe,oo,m16);
        Dot(delta2,dy2e,dy2o,oe,oo,m);
    }
    RegTensor<float> p20, ds20;
    {
        RegTensor<float> l;
        LoadAlign<float,LoadDist::DIST_BRC_B32>(l,lse+2);
        Dot(p20,q2e,q2o,k0e,k0o,m);
        Muls(p20,p20,scale,m);
        Sub(p20,p20,l,m);
        Exp(p20,p20,m);
        Dot(ds20,dy2e,dy2o,v0e,v0o,m);
        Sub(ds20,ds20,delta2,m);
        Mul(ds20,ds20,p20,m);
        Muls(ds20,ds20,scale,m);
    }
    RegTensor<float> p21, ds21;
    {
        RegTensor<float> l;
        LoadAlign<float,LoadDist::DIST_BRC_B32>(l,lse+2);
        Dot(p21,q2e,q2o,k1e,k1o,m);
        Muls(p21,p21,scale,m);
        Sub(p21,p21,l,m);
        Exp(p21,p21,m);
        Dot(ds21,dy2e,dy2o,v1e,v1o,m);
        Sub(ds21,ds21,delta2,m);
        Mul(ds21,ds21,p21,m);
        Muls(ds21,ds21,scale,m);
    }
    RegTensor<float> p22, ds22;
    {
        RegTensor<float> l;
        LoadAlign<float,LoadDist::DIST_BRC_B32>(l,lse+2);
        Dot(p22,q2e,q2o,k2e,k2o,m);
        Muls(p22,p22,scale,m);
        Sub(p22,p22,l,m);
        Exp(p22,p22,m);
        Dot(ds22,dy2e,dy2o,v2e,v2o,m);
        Sub(ds22,ds22,delta2,m);
        Mul(ds22,ds22,p22,m);
        Muls(ds22,ds22,scale,m);
    }
    {
        RegTensor<float> e,o,te,to;
        Mul(e,k0e,ds00,m);
        Mul(o,k0o,ds00,m);
        StoreRow(dq+0,e,o,m16);
    }
    {
        RegTensor<float> e,o,te,to;
        Mul(e,q0e,ds00,m);
        Mul(o,q0o,ds00,m);
        Mul(te,q1e,ds10,m);
        Mul(to,q1o,ds10,m);
        Add(e,e,te,m);
        Add(o,o,to,m);
        Mul(te,q2e,ds20,m);
        Mul(to,q2o,ds20,m);
        Add(e,e,te,m);
        Add(o,o,to,m);
        StoreRow(dk+0,e,o,m16);
    }
    {
        RegTensor<float> e,o,te,to;
        Mul(e,dy0e,p00,m);
        Mul(o,dy0o,p00,m);
        Mul(te,dy1e,p10,m);
        Mul(to,dy1o,p10,m);
        Add(e,e,te,m);
        Add(o,o,to,m);
        Mul(te,dy2e,p20,m);
        Mul(to,dy2o,p20,m);
        Add(e,e,te,m);
        Add(o,o,to,m);
        StoreRow(dv+0,e,o,m16);
    }
    {
        RegTensor<float> e,o,te,to;
        Mul(e,k0e,ds10,m);
        Mul(o,k0o,ds10,m);
        Mul(te,k1e,ds11,m);
        Mul(to,k1o,ds11,m);
        Add(e,e,te,m);
        Add(o,o,to,m);
        StoreRow(dq+128,e,o,m16);
    }
    {
        RegTensor<float> e,o,te,to;
        Mul(e,q1e,ds11,m);
        Mul(o,q1o,ds11,m);
        Mul(te,q2e,ds21,m);
        Mul(to,q2o,ds21,m);
        Add(e,e,te,m);
        Add(o,o,to,m);
        StoreRow(dk+128,e,o,m16);
    }
    {
        RegTensor<float> e,o,te,to;
        Mul(e,dy1e,p11,m);
        Mul(o,dy1o,p11,m);
        Mul(te,dy2e,p21,m);
        Mul(to,dy2o,p21,m);
        Add(e,e,te,m);
        Add(o,o,to,m);
        StoreRow(dv+128,e,o,m16);
    }
    {
        RegTensor<float> e,o,te,to;
        Mul(e,k0e,ds20,m);
        Mul(o,k0o,ds20,m);
        Mul(te,k1e,ds21,m);
        Mul(to,k1o,ds21,m);
        Add(e,e,te,m);
        Add(o,o,to,m);
        Mul(te,k2e,ds22,m);
        Mul(to,k2o,ds22,m);
        Add(e,e,te,m);
        Add(o,o,to,m);
        StoreRow(dq+256,e,o,m16);
    }
    {
        RegTensor<float> e,o,te,to;
        Mul(e,q2e,ds22,m);
        Mul(o,q2o,ds22,m);
        StoreRow(dk+256,e,o,m16);
    }
    {
        RegTensor<float> e,o,te,to;
        Mul(e,dy2e,p22,m);
        Mul(o,dy2o,p22,m);
        StoreRow(dv+256,e,o,m16);
    }
}
template<class T>
CATLASS_DEVICE void Run(FAGKernelParams const& p, const __gm__ FAGTiling950::FAGTilingData* tiling)
{
    using namespace AscendC;
    Catlass::Arch::Resource<Catlass::Arch::Ascend950> resource;
    GlobalTensor<T> q,k,v,dy,o,dq,dk,dv;
    GlobalTensor<float> lse;
    q.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(p.q));
    k.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(p.k));
    v.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(p.v));
    dy.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(p.dout));
    o.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(p.out));
    dq.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(p.dq));
    dk.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(p.dk));
    dv.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(p.dv));
    lse.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(p.softmaxLse));

    const uint32_t heads=tiling->qHeadNum;
    const uint32_t total=tiling->batch*heads;
    const uint32_t columns=tiling->totalQ;
    const float scale=tiling->scaleValue;
    const uint32_t rank=GetBlockIdx();
    const uint32_t cores=tiling->usedCoreNum*GetSubBlockNum();
    constexpr uint32_t n=3*128;
    constexpr uint32_t slotBytes=8*n*sizeof(T)+32;
    const uint32_t gap=(heads-1)*128*sizeof(T);
    DataCopyExtParams in{3,128*sizeof(T),gap,0,0},out{3,128*sizeof(T),0,gap,0};
    DataCopyExtParams stat{1,3*sizeof(float),0,0,0};
    DataCopyPadExtParams<T> pad{false,0,0,0};
    DataCopyPadExtParams<float> spad{false,0,0,0};
    for (uint32_t i=0;i<2;++i) {
        SetFlag<HardEvent::V_MTE2>(i);
        SetFlag<HardEvent::MTE3_V>(i);
    }
    for (uint32_t task=rank,i=0;task<total;task+=cores,++i) {
        const uint32_t head=task%heads;
        const uint32_t batch=task/heads;
        const uint64_t addr=(static_cast<uint64_t>(batch)*3*heads+head)*128;
        const uint32_t ev=i&1U;
        auto buf=resource.ubBuf.template GetBufferByByte<T>(ev*slotBytes);
        auto stats=resource.ubBuf.template GetBufferByByte<float>(ev*slotBytes+8*n*sizeof(T));
        WaitFlag<HardEvent::V_MTE2>(ev);
        DataCopyPad(buf[0*n],q[addr],in,pad);
        DataCopyPad(buf[1*n],k[addr],in,pad);
        DataCopyPad(buf[2*n],v[addr],in,pad);
        DataCopyPad(buf[3*n],dy[addr],in,pad);
        DataCopyPad(buf[4*n],o[addr],in,pad);
        DataCopyPad(stats,lse[static_cast<uint64_t>(head)*columns+batch*3U],stat,spad);
        SetFlag<HardEvent::MTE2_V>(ev);
        WaitFlag<HardEvent::MTE2_V>(ev);
        WaitFlag<HardEvent::MTE3_V>(ev);
        auto x=reinterpret_cast<__ubuf__ T*>(buf.GetPhyAddr());
        Backward(x,x+n,x+2*n,x+3*n,x+4*n,reinterpret_cast<__ubuf__ float*>(stats.GetPhyAddr()),
                 x+5*n,x+6*n,x+7*n,scale);
        SetFlag<HardEvent::V_MTE2>(ev);
        SetFlag<HardEvent::V_MTE3>(ev);
        WaitFlag<HardEvent::V_MTE3>(ev);
        DataCopyPad(dq[addr],buf[5*n],out);
        DataCopyPad(dk[addr],buf[6*n],out);
        DataCopyPad(dv[addr],buf[7*n],out);
        SetFlag<HardEvent::MTE3_V>(ev);
    }
    for (uint32_t i=0;i<2;++i) {
        WaitFlag<HardEvent::V_MTE2>(i);
        WaitFlag<HardEvent::MTE3_V>(i);
    }
}
} // namespace FagTiny
#endif
#endif
