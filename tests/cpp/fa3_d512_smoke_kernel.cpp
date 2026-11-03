#include "d512/kernel.hpp"
template <class T, Format F, CacheMode C = CacheMode::normalCache, PageShape S = PageShape::normalShape,
          MaskCategory M = MaskCategory::NO_MASK>
void launch_d512(void** b, void* stream, int kvLength, int cores)
{
    FAInferD512<T, float, F, F, C, S, M, CacheLayout::nd, false, false><<<cores, nullptr, stream>>>(
        (GM_ADDR)b[0], (GM_ADDR)b[1], (GM_ADDR)b[2], (GM_ADDR)b[8], (GM_ADDR)b[9], (GM_ADDR)b[3], nullptr,
        (GM_ADDR)b[6], (GM_ADDR)b[7], nullptr, nullptr, (GM_ADDR)b[4], (GM_ADDR)b[5], 0.0441941738f, 0.0f,
        (kvLength + 127) / 128, (kvLength + 127) / 128, kvLength);
}

void check_d512(void** b, void* stream, int kvLength, bool bf16, bool bsnd, int cores, bool paged, bool causal)
{
    if (paged) {
        launch_d512<half, Format::TND, CacheMode::pagedCache, PageShape::BnBsND>(b, stream, kvLength, cores);
    } else if (causal) {
        launch_d512<half, Format::TND, CacheMode::normalCache, PageShape::normalShape, MaskCategory::MASK_CAUSAL>(
            b, stream, kvLength, cores);
    } else if (bsnd) {
        if (bf16)
            launch_d512<bfloat16_t, Format::BSND>(b, stream, kvLength, cores);
        else
            launch_d512<half, Format::BSND>(b, stream, kvLength, cores);
    } else {
        if (bf16)
            launch_d512<bfloat16_t, Format::TND>(b, stream, kvLength, cores);
        else
            launch_d512<half, Format::TND>(b, stream, kvLength, cores);
    }
}
