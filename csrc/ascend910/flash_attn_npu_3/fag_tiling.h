#ifndef FAG_TILING_H
#define FAG_TILING_H

#include <array>
#include <cstdint>

namespace FAGTiling {
constexpr uint32_t INITIAL_S1_SPLIT_NUM = 128; // to avoid repeat max value 255
constexpr uint32_t INITIAL_S2_SPLIT_NUM = 64;
constexpr uint32_t SAMEAB_S1_BASE = 512;
constexpr uint32_t SAMEAB_S2_BASE = 512;
constexpr uint32_t INPUT_ALIGN = 16;
constexpr uint32_t EMPTY_TENSOR = 0;
constexpr uint32_t NORMAL_TENSOR = 1;
constexpr uint32_t BIT_NUMS = 8;
constexpr uint32_t FP16_BYTES = 2;
constexpr uint32_t FP16_BLOCK_NUMS = 16;
constexpr uint32_t FP32_BYTES = 4;
constexpr uint32_t FP32_BLOCK_NUMS = 8;
constexpr uint32_t BASIC_BLOCK_MULTIPLE = 15;
constexpr uint32_t SHAPE_INFO = 32;
constexpr uint32_t BYTE_BLOCK = 32;     // 32 B in block
constexpr int64_t GM_ALIGN = 512;
constexpr uint32_t MUL_CORE_SYNC_BUFFER = 64 * 1024;
constexpr uint32_t BOOL_BLOCK_NUMS = 32;
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t WORKSPACE_NUM_ALIGN = 256;
constexpr uint32_t POST_COEX_NODE = 3;
constexpr uint32_t WORKSPACE_BUFFER = 20 * 1024 * 1024;
constexpr uint32_t SOFTMAX_REDUCE_SIZE = 8;
constexpr uint32_t MATMUL_INPUT_NUM = 2;
constexpr uint32_t MATMUL_SIZE = 8 * 1024;
constexpr uint32_t TOTAL_BLOCK_PIPELINE = 64;
constexpr uint32_t SAMEAB_S1_256 = 256;
constexpr uint32_t DB_NUM = 2;
constexpr int64_t VEC_SPLIT_NUM = 3;
constexpr int64_t TOTAL_SIZE = 189 * 1024;
constexpr float HALF = 0.5f;

enum class MaskType {
    NO_MASK = 0,
    MASK_CAUSUAL = 1,
    MASK_BAND = 2
};

enum AttenMaskCompressMode {
    NO_COMPRESS_MODE = 0,
    CAUSAL_COMPRESS_MODE = 1,
    BAND_COMPRESS_MODE = 2
};

constexpr int64_t ATTEN_MASK_COMPRESS_DIM = 2048;

struct FAGInfo {
    float scaleValue;
    float softcapValue;
    float keepProb;
    int32_t maskType = 0;
    int32_t layout = 0;

    int64_t batch = 0;
    int64_t qSeqlen = 0;
    int64_t qHeadNum = 0;
    int64_t qkHeadDim = 0;
    int64_t kvSeqlen = 0;
    int64_t kvHeadNum = 0;
    int64_t vHeadDim = 0;
    int64_t window_size_left;
    int64_t window_size_right;
    int32_t *qSeqlenList{nullptr};
    int32_t *kvSeqlenList{nullptr};
    bool isDeterministic = false;
    int64_t alibiSlopesBatchStride = 0;
};
} // namespace FAGTiling

#endif
