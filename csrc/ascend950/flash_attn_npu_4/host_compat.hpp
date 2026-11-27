// Copyright (c) 2026, Minghua Shen.
//
// Host-build compatibility shim for the shared ASC/CATLASS headers.
//
// Since torch 2.14 the PyTorch/torch_npu headers require C++20, which the
// bisheng ASC frontend (-x asc) cannot parse (libstdc++ C++20 constrained
// overloads trip its overload-redeclaration check). Host translation units -
// the torch-facing flash_api.cpp files - are therefore compiled by the plain
// C++ (host) frontend, while kernel translation units stay on the ASC
// frontend. Both sides include headers such as kernel_common.hpp that use a
// small amount of ASC vocabulary (GM_ADDR, __aicore__, __global__,
// __forceinline__, AscendC::Std::min). This shim provides that vocabulary so
// the shared headers parse unchanged under the host frontend; the build
// force-includes it (-include) into host TUs, and it is inert under the ASC
// frontend. Binary layout is preserved: pointers stay plain 8-byte pointers,
// and ASC-only attributes are dropped.

#if !defined(__CCE__)
#ifndef FLASH_ATTN_NPU_HOST_COMPAT_HPP
#define FLASH_ATTN_NPU_HOST_COMPAT_HPP

#include <cstdint>
#include <cstddef> // catlass tla/type_traits.hpp uses unqualified size_t
#include <algorithm>

// adv_api/kernel_tiling.h is host-clean (plain structs) and provides
// SoftMaxTiling used by the softmax tiling host math.
#include "adv_api/kernel_tiling.h"

// ASC function attributes: meaningless on the host; drop them. Guarded so an
// ASC-side accidental include is a no-op (the compiler defines them itself).
#ifndef __aicore__
#define __aicore__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif

// CANN: #define GM_ADDR __gm__ uint8_t*  (kernel_utils_macros.h). The address
// space qualifier is irrelevant on the host; the representation is identical.
#ifndef __gm__
#define __gm__
#endif
#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

// KernelCommon::GetStackSeqTile (kernel_common.hpp) uses AscendC::Std::min on
// both sides.
namespace AscendC {
namespace Std {
using std::max;
using std::min;
} // namespace Std
} // namespace AscendC

#endif // FLASH_ATTN_NPU_HOST_COMPAT_HPP
#endif // !__CCE__
