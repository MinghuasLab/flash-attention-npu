#pragma once
#include "../fag_block.h"
namespace Catlass::Epilogue {
template <uint32_t L, bool D, bool M, bool S, bool A>
struct EpilogueAtlasA2MlaVec : EpilogueAtlasA2SameAbVec<L, D, M, S, A> {};
} // namespace Catlass::Epilogue
