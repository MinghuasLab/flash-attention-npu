#include <acl/acl.h>
#include <cstdio>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include "kernel_common.hpp"
#include "tilingdata.h"
#include "d512/config.hpp"
void check_d512(void**, void*, int, bool, bool, int, bool, bool);
#define CHECK(x)                                                                                                       \
    do {                                                                                                               \
        auto e = (x);                                                                                                  \
        if (e != ACL_SUCCESS) {                                                                                        \
            std::printf("ACL failure %d at %d\n", int(e), __LINE__);                                                   \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)
int main(int argc, char** argv)
{
    setbuf(stdout, nullptr);
    std::puts("smoke init");
    int kvLength = argc > 1 ? std::atoi(argv[1]) : 128;
    int qLength = argc > 2 ? std::atoi(argv[2]) : 128;
    const bool bf16 = argc > 3 && std::atoi(argv[3]) != 0;
    const bool bsnd = argc > 4 && std::atoi(argv[4]) != 0;
    const bool varied = argc > 10 && std::atoi(argv[10]) != 0;
    const int batch = (bsnd || varied) ? 2 : 1;
    const int cores = argc > 5 ? std::atoi(argv[5]) : 1;
    const bool paged = argc > 8 && std::atoi(argv[8]) != 0;
    const bool causal = argc > 9 && std::atoi(argv[9]) != 0;
    const int heads = argc > 6 ? std::atoi(argv[6]) : 1;
    const int kvHeads = argc > 7 ? std::atoi(argv[7]) : 1;
    const size_t workspaceBytes = static_cast<size_t>(cores) * flash_attn::d512::LO_CORE_BYTES;
    const uint16_t one = bf16 ? 0x3f80 : 0x3c00;
    const int blocks = (kvLength + 127) / 128;
    const int allocatedKv = paged ? blocks * 128 : kvLength;
    const int totalQ = batch * qLength - (varied ? 1 : 0);
    const int totalKv = batch * allocatedKv - (varied ? 1 : 0);
    const size_t kvBytes = static_cast<size_t>(totalKv) * kvHeads * 512 * 2;
    const char* deviceEnv = std::getenv("FA_D512_TEST_DEVICE");
    const int device = deviceEnv ? std::atoi(deviceEnv) : 0;
    CHECK(aclInit(nullptr));
    CHECK(aclrtSetDevice(device));
    aclrtStream stream;
    CHECK(aclrtCreateStream(&stream));
    const size_t elems = static_cast<size_t>(totalQ) * heads * 512, bytes = elems * 2;
    std::vector<uint16_t> zero(elems, 0), zeroK(totalKv * kvHeads * 512, 0), ones(totalKv * kvHeads * 512, one),
        out(elems, 0);
    const char* randomInput = std::getenv("FA_D512_RANDOM_INPUT");
    const char* randomOutput = std::getenv("FA_D512_RANDOM_OUTPUT");
    if (randomInput && randomOutput) {
        std::ifstream input(randomInput, std::ios::binary);
        input.read(reinterpret_cast<char*>(zero.data()), bytes);
        input.read(reinterpret_cast<char*>(zeroK.data()), kvBytes);
        input.read(reinterpret_cast<char*>(ones.data()), kvBytes);
        if (!input) {
            std::fprintf(stderr, "Failed to read random input\n");
            return 1;
        }
    }
    void *q, *k, *v, *o, *ws, *td, *seq, *seqKv, *mask = nullptr, *table = nullptr;
    CHECK(aclrtMalloc(&q, bytes, ACL_MEM_MALLOC_NORMAL_ONLY));
    CHECK(aclrtMalloc(&k, kvBytes, ACL_MEM_MALLOC_NORMAL_ONLY));
    CHECK(aclrtMalloc(&v, kvBytes, ACL_MEM_MALLOC_NORMAL_ONLY));
    CHECK(aclrtMalloc(&o, bytes, ACL_MEM_MALLOC_NORMAL_ONLY));
    CHECK(aclrtMalloc(&ws, workspaceBytes, ACL_MEM_MALLOC_NORMAL_ONLY));
    CHECK(aclrtMalloc(&td, sizeof(FAInferTilingData), ACL_MEM_MALLOC_NORMAL_ONLY));
    CHECK(aclrtMalloc(&seq, 3 * sizeof(int32_t), ACL_MEM_MALLOC_NORMAL_ONLY));
    CHECK(aclrtMemcpy(q, bytes, zero.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK(aclrtMemcpy(k, kvBytes, zeroK.data(), kvBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK(aclrtMemcpy(v, kvBytes, ones.data(), kvBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK(aclrtMemset(o, bytes, 0xff, bytes));
    CHECK(aclrtMemset(ws, workspaceBytes, 0, workspaceBytes));
    int32_t seqlens[3] = {0, qLength, totalQ};
    CHECK(aclrtMemcpy(seq, sizeof(seqlens), seqlens, sizeof(seqlens), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK(aclrtMalloc(&seqKv, 3 * sizeof(int32_t), ACL_MEM_MALLOC_NORMAL_ONLY));
    int32_t kvSeq[3] = {(bsnd || paged) ? kvLength : 0, kvLength, totalKv};
    CHECK(aclrtMemcpy(seqKv, sizeof(kvSeq), kvSeq, sizeof(kvSeq), ACL_MEMCPY_HOST_TO_DEVICE));
    FAInferTilingData t{};
    t.batch = batch;
    t.numHeads = heads;
    t.kvHeads = kvHeads;
    t.embeddingSize = 512;
    t.embeddingSizeV = 512;
    t.maxQSeqlen = qLength;
    const int groupSize = heads / kvHeads;
    const int headTile = flash_attn::d512::QHeadTile(qLength, groupSize);
    t.firstBatchTaskNum = ((qLength + 127) / 128) * ((groupSize + headTile - 1) / headTile) * kvHeads;
    t.totalTaskNum = t.firstBatchTaskNum * batch;
    if (varied) {
        const int lastQ = qLength - 1;
        const int lastHeadTile = flash_attn::d512::QHeadTile(lastQ, groupSize);
        t.totalTaskNum =
            t.firstBatchTaskNum + ((lastQ + 127) / 128) * ((groupSize + lastHeadTile - 1) / lastHeadTile) * kvHeads;
    }
    t.qSeqlenAligned = qLength;
    t.kvSeqlenAligned = kvLength;
    t.blockSize = 128;
    t.windowSizeLeft = -1;
    t.windowSizeRight = -1;
    flash_attn::d512::ApplyTiling(t);
    CHECK(aclrtMemcpy(td, sizeof(t), &t, sizeof(t), ACL_MEMCPY_HOST_TO_DEVICE));
    if (paged) {
        std::vector<int32_t> blockTable(blocks);
        for (int i = 0; i < blocks; ++i)
            blockTable[i] = blocks - 1 - i;
        CHECK(aclrtMalloc(&table, blocks * sizeof(int32_t), ACL_MEM_MALLOC_NORMAL_ONLY));
        CHECK(aclrtMemcpy(table, blocks * sizeof(int32_t), blockTable.data(), blocks * sizeof(int32_t),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    }
    if (causal) {
        std::vector<uint8_t> triangular(2048 * 2048);
        for (int r = 0; r < 2048; ++r)
            for (int c = r + 1; c < 2048; ++c)
                triangular[r * 2048 + c] = 1;
        CHECK(aclrtMalloc(&mask, triangular.size(), ACL_MEM_MALLOC_NORMAL_ONLY));
        CHECK(aclrtMemcpy(mask, triangular.size(), triangular.data(), triangular.size(), ACL_MEMCPY_HOST_TO_DEVICE));
    }
    void* buffers[] = {q, k, v, o, ws, td, seq, seqKv, mask, table};
    std::puts("smoke launch");
    check_d512(buffers, stream, kvLength, bf16, bsnd, cores, paged, causal);
    std::puts("smoke synchronize");
    CHECK(aclrtSynchronizeStream(stream));
    CHECK(aclrtMemcpy(out.data(), bytes, o, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<float> lo(128 * 512);
    CHECK(aclrtMemcpy(lo.data(), lo.size() * 4, ws, lo.size() * 4, ACL_MEMCPY_DEVICE_TO_HOST));
    for (int r : {0, 56, 57, 63, 64, 120, 121, 127}) {
        if (r < qLength)
            std::printf("LO row %d: %g %g O=0x%x\n", r, lo[r * 512], lo[r * 512 + 511], out[r * 512]);
    }
    unsigned bad = 0;
    if (randomInput && randomOutput) {
        std::ofstream output(randomOutput, std::ios::binary);
        output.write(reinterpret_cast<const char*>(out.data()), bytes);
        if (!output)
            return 1;
    } else
        for (size_t i = 0; i < elems; ++i)
            if (out[i] != one) {
                if (bad < 5)
                    std::printf("bad[%zu]=0x%x\n", i, out[i]);
                ++bad;
            }
    if (!randomInput)
        std::printf("D512 uniform smoke: bad=%u/%zu\n", bad, elems);
    for (void* p : {q, k, v, o, ws, td, seq, seqKv})
        CHECK(aclrtFree(p));
    if (mask)
        CHECK(aclrtFree(mask));
    if (table)
        CHECK(aclrtFree(table));
    CHECK(aclrtDestroyStream(stream));
    CHECK(aclrtResetDevice(device));
    CHECK(aclFinalize());
    return bad ? 2 : 0;
}
