// Copyright (c) 2026, Minghua Shen.
//
// torch-free bridge to torch_npu's aicpu op-tracking entry.
//
// ASC (device) translation units must not include torch/torch_npu headers:
// since torch 2.14 those headers require C++20, which the bisheng ASC
// frontend cannot parse. Device TUs that need to wrap their kernel launch in
// the aicpu op-tracking context call fa_bridge::run_op_api_v2() instead; the
// definition lives in the torch-facing host TU of the same extension
// (flash_api.cpp), which includes the real
// torch_npu/csrc/framework/OpCommand.h.

#ifndef FLASH_ATTN_NPU_OP_COMMAND_BRIDGE_HPP
#define FLASH_ATTN_NPU_OP_COMMAND_BRIDGE_HPP

#include <functional>

namespace fa_bridge {
// Equivalent of at_npu::native::OpCommand::RunOpApiV2(op_name, func).
// Implemented once per extension in the host TU; keep the argument types in
// sync with PROC_FUNC = std::function<int()> in torch_npu's OpCommand.h.
void run_op_api_v2(const char* op_name, const std::function<int()>& func);
} // namespace fa_bridge

#endif // FLASH_ATTN_NPU_OP_COMMAND_BRIDGE_HPP
