/** Copyright (c) 2026 Huawei Technologies Co., Ltd. CANN Open Software License Agreement Version 2.0. */
#ifndef FA950_FWD_RESOURCE_HPP
#define FA950_FWD_RESOURCE_HPP
#include "catlass/arch/resource.hpp"

namespace Catlass::Arch {
// CO1 and C2PIPE2GM belong to the Cube. On AIV, InitBuffer does not
// initialize these TBuf descriptors, so calling Get() there reads private
// storage that has never been written. AIV never consumes either buffer.
template <AscendC::TPosition Position, uint32_t Bytes>
struct ForwardCubeBuffer950 : LocalTensorBufferBase {
    __aicore__ inline ForwardCubeBuffer950()
    {
#ifdef __DAV_CUBE__
        AscendC::TBuf<Position> buffer;
        GetTPipePtr()->InitBuffer(buffer, Bytes);
        tensor = buffer.template Get<uint8_t>();
#endif
    }
};

struct ForwardResource950 {
    AscendC::TPipe pipe;
    LocalTensorBuffer<Ascend950, AscendC::TPosition::A1> l1Buf;
    LocalTensorBuffer<Ascend950, AscendC::TPosition::A2> l0ABuf;
    LocalTensorBuffer<Ascend950, AscendC::TPosition::B2> l0BBuf;
    LocalTensorBuffer<Ascend950, AscendC::TPosition::C2> btBuf;
    ForwardCubeBuffer950<AscendC::TPosition::CO1, Ascend950::L0C_SIZE> l0CBuf;
    LocalTensorBuffer<Ascend950, AscendC::TPosition::VECCALC> ubBuf;
    ForwardCubeBuffer950<AscendC::TPosition::C2PIPE2GM, Ascend950::FIXBUF_SIZE> fpBuf;

    __aicore__ inline ForwardResource950() { pipe.Destroy(); }
};
} // namespace Catlass::Arch
#endif
