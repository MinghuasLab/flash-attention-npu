// Standalone GM->UB probe: vary rows/request while holding useful bytes fixed.
// Arguments: rows heads tokens_per_vector serialized device (default device 1).
// This intentionally omits paging, scaling and attention. It measures the
// transfer primitive's granularity, not end-to-end attention performance.
#include "kernel_operator.h"
#include "acl/acl.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" __global__ __aicore__ void KvReadGranularity950(
    GM_ADDR source, GM_ADDR output, uint32_t rows, uint32_t heads,
    uint32_t tokens, uint32_t serialized)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scratch;
    pipe.InitBuffer(scratch, 2U * 512U * 192U);
    auto ub = scratch.Get<uint8_t>();
    AscendC::GlobalTensor<uint8_t> src, dst;
    src.SetGlobalBuffer(source);
    dst.SetGlobalBuffer(output);
    const uint32_t core = AscendC::GetBlockIdx();
    const uint64_t base = (uint64_t(core / heads) * tokens * heads + core % heads) * 192U;
    const AscendC::DataCopyExtParams copy{static_cast<uint16_t>(rows), 192U, (heads - 1U) * 192U, 0, 0};
    const AscendC::DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
    uint32_t lastSlot = 0;
    for (uint32_t token = 0, iteration = 0; token < tokens; token += rows, ++iteration) {
        lastSlot = iteration % 2U;
        AscendC::DataCopyPad(ub[lastSlot * 512U * 192U], src[base + uint64_t(token) * heads * 192U], copy, pad);
        if (serialized != 0U) {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        }
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
    AscendC::DataCopy(dst[core * 192U], ub[lastSlot * 512U * 192U], 192U);
}

static void Check(aclError error)
{
    if (error != ACL_SUCCESS) {
        std::fprintf(stderr, "ACL error %d\n", int(error));
        std::exit(2);
    }
}

int main(int argc, char** argv)
{
    const uint32_t rows = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 64U;
    const uint32_t heads = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 4U;
    const uint32_t tokens = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 8192U;
    const uint32_t serialized = argc > 4 ? std::strtoul(argv[4], nullptr, 10) : 1U;
    const int device = argc > 5 ? std::atoi(argv[5]) : 1;
    if (rows < 1 || rows > 512 || tokens < rows || tokens % rows ||
        (heads != 1 && heads != 2 && heads != 4)) { return 2; }
    const size_t bytes = size_t(64) * tokens * 192U;
    std::vector<uint8_t> host(bytes), result(64U * 192U);
    for (size_t i = 0; i < bytes; ++i) { host[i] = uint8_t(i ^ (i >> 8) ^ (i >> 16)); }
    Check(aclInit(nullptr));
    Check(aclrtSetDevice(device));
    aclrtStream stream;
    Check(aclrtCreateStream(&stream));
    void *source, *output;
    Check(aclrtMalloc(&source, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    Check(aclrtMalloc(&output, result.size(), ACL_MEM_MALLOC_HUGE_FIRST));
    Check(aclrtMemcpy(source, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    auto run = [&]() {
        KvReadGranularity950<<<64, nullptr, stream>>>(static_cast<uint8_t*>(source),
            static_cast<uint8_t*>(output), rows, heads, tokens, serialized);
    };
    for (int i = 0; i < 5; ++i) { run(); }
    Check(aclrtSynchronizeStream(stream));
    aclrtEvent start, end;
    Check(aclrtCreateEventWithFlag(&start, ACL_EVENT_TIME_LINE));
    Check(aclrtCreateEventWithFlag(&end, ACL_EVENT_TIME_LINE));
    std::vector<float> samples;
    for (int i = 0; i < 7; ++i) {
        Check(aclrtRecordEvent(start, stream));
        for (int j = 0; j < 20; ++j) { run(); }
        Check(aclrtRecordEvent(end, stream));
        Check(aclrtSynchronizeStream(stream));
        float ms;
        Check(aclrtEventElapsedTime(&ms, start, end));
        samples.push_back(ms * 1000.0f / 20.0f);
    }
    Check(aclrtMemcpy(result.data(), result.size(), output, result.size(), ACL_MEMCPY_DEVICE_TO_HOST));
    for (uint32_t core = 0; core < 64U; ++core) {
        const size_t offset = (size_t(core / heads) * tokens * heads + core % heads +
            size_t(tokens - rows) * heads) * 192U;
        for (uint32_t d = 0; d < 192U; ++d) {
            if (result[core * 192U + d] != host[offset + d]) {
                std::fprintf(stderr, "Copy mismatch at core %u dim %u\n", core, d);
                return 3;
            }
        }
    }
    std::sort(samples.begin(), samples.end());
    std::printf("{\"rows\":%u,\"heads\":%u,\"tokens_per_vector\":%u,\"serialized\":%u,"
                "\"useful_bytes\":%zu,\"median_us\":%.6f,\"useful_GB_s\":%.3f}\n",
        rows, heads, tokens, serialized, bytes, samples[3], bytes / (samples[3] * 1000.0));
    Check(aclrtDestroyEvent(start));
    Check(aclrtDestroyEvent(end));
    Check(aclrtFree(source));
    Check(aclrtFree(output));
    Check(aclrtDestroyStream(stream));
    Check(aclrtResetDevice(device));
    Check(aclFinalize());
    return 0;
}
