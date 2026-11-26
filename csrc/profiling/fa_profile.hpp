/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0.
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Modified by Minghua Shen, 2026.
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <initializer_list>
#include <locale>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <limits>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

#include <ATen/ATen.h>

#include "acl/acl.h"
#include "acl/acl_prof.h"
#include "profiling/aprof_pub.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"

namespace fa_profile __attribute__((visibility("hidden"))) {

    constexpr unsigned kShapes = 1;
    constexpr unsigned kAttributes = 2;
    inline std::atomic<unsigned> mode{0};
    inline std::atomic<uint64_t> sequence{0};

    inline bool AttributesRequested(const MsprofCommandHandle& command)
    {
        if ((command.profSwitch & PROF_MSPROFTX_MASK) != 0) {
            return true;
        }

        // msprof --application passes its switches through this configuration.
        // In-process aclprof sessions use the callback bit, which reflects each restart.
        const char* config = std::getenv("PROFILER_SAMPLECONFIG");
        if (config == nullptr) {
            return false;
        }
        const std::string_view options(config);
        constexpr std::string_view key = "\"msproftx\"";
        constexpr std::string_view whitespace = " \t\r\n";
        const auto field = options.find(key);
        if (field == std::string_view::npos) {
            return false;
        }
        const auto colon = options.find_first_not_of(whitespace, field + key.size());
        if (colon == std::string_view::npos || options[colon] != ':') {
            return false;
        }
        const auto value = options.find_first_not_of(whitespace, colon + 1);
        return value != std::string_view::npos && options.substr(value, 4) == "\"on\"";
    }

    inline int32_t UpdateProfilingState(uint32_t control, void* data, uint32_t length)
    {
        if (control != PROF_CTRL_SWITCH || data == nullptr || length != sizeof(MsprofCommandHandle)) {
            return 0;
        }

        const auto& command = *static_cast<const MsprofCommandHandle*>(data);
        switch (command.type) {
        case PROF_COMMANDHANDLE_TYPE_START: {
            const bool shapes =
                (command.profSwitch & PROF_TASK_TIME_MASK) != 0 && (command.profSwitch & PROF_TASK_TIME_L1_MASK) != 0;
            const bool attributes = shapes && AttributesRequested(command);
            mode.store(shapes ? kShapes | (attributes ? kAttributes : 0U) : 0U, std::memory_order_relaxed);
            break;
        }
        case PROF_COMMANDHANDLE_TYPE_STOP:
        case PROF_COMMANDHANDLE_TYPE_FINALIZE:
            mode.store(0, std::memory_order_relaxed);
            break;
        default:
            break;
        }
        return 0;
    }

    inline void RegisterProfilingCallback()
    {
        // Match the ACLNN reporting module. CANN maps l0 + aicore-shape to the
        // L1 tensor-info bit for this module and accepts multiple callbacks.
        constexpr uint32_t nnopbase_module_id = 61;
        static const int32_t status = MsprofRegisterCallback(nnopbase_module_id, UpdateProfilingState);
        (void)status;
    }

    inline bool IsEnabled()
    {
        return (mode.load(std::memory_order_relaxed) & kShapes) != 0;
    }

    template <typename T>
    const at::Tensor* Tensor(const std::optional<T>& value)
    {
        return value ? &*value : nullptr;
    }

    struct InputTensor {
        const char* name;
        const at::Tensor* tensor;
        const char* loweredName = nullptr;
    };

    inline void ReportAttributes(const char* kernel, uint64_t begin, const std::string& attributes)
    {
        if (attributes.empty() || (mode.load(std::memory_order_relaxed) & kAttributes) == 0) {
            return;
        }
        // The runtime accepts up to 126 message bytes per stamp.
        constexpr size_t maxMessageLength = 126;
        std::string message =
            std::string(kernel) + " launch=" + std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)) + ' ';
        message.reserve(maxMessageLength);
        const size_t prefix = message.size();
        const size_t capacity = maxMessageLength - prefix;
        const auto fields = "begin=" + std::to_string(begin) + attributes;
        std::string_view pending(fields);
        auto* stamp = aclprofCreateStamp();
        if (!stamp) {
            return;
        }
        while (!pending.empty()) {
            size_t count = pending.size();
            if (count > capacity) {
                count = pending.rfind(' ', capacity);
                if (count == std::string_view::npos) {
                    break;
                }
            }
            message.resize(prefix);
            message.append(pending.substr(0, count));
            if (aclprofSetStampTraceMessage(stamp, message.c_str(), static_cast<uint32_t>(message.size())) ==
                ACL_SUCCESS) {
                (void)aclprofMark(stamp);
            }
            pending.remove_prefix(count == pending.size() ? count : count + 1);
        }
        aclprofDestroyStamp(stamp);
    }

    inline void Window(std::ostream & attributes, bool causal, int64_t left, int64_t right)
    {
        attributes << " causal=" << causal << " window_size=[" << left << ',' << right << ']';
    }

    inline aclDataType ProfileDataType(at::ScalarType type)
    {
        switch (type) {
        case at::kHalf:
            return ACL_FLOAT16;
        case at::kBFloat16:
            return ACL_BF16;
        case at::ScalarType::Float8_e4m3fn:
            return ACL_FLOAT8_E4M3FN;
        case at::kFloat:
            return ACL_FLOAT;
        case at::kInt:
            return ACL_INT32;
        case at::kByte:
            return ACL_UINT8;
        case at::kLong:
            return ACL_INT64;
        case at::kChar:
            return ACL_INT8;
        case at::kBool:
            return ACL_BOOL;
        default:
            return ACL_DT_UNDEFINED;
        }
    }

    class TensorInfo {
      public:
        TensorInfo(std::initializer_list<const at::Tensor*> inputs, std::initializer_list<const at::Tensor*> outputs)
        {
            if (IsEnabled()) {
                tensors_.reserve(inputs.size() + outputs.size());
                Append(inputs, MSPROF_GE_TENSOR_TYPE_INPUT);
                Append(outputs, MSPROF_GE_TENSOR_TYPE_OUTPUT);
            }
        }

        template <typename Writer>
        TensorInfo(const char* kernel, std::initializer_list<InputTensor> inputs,
                   std::initializer_list<const at::Tensor*> outputs, Writer&& write)
        {
            const unsigned active = mode.load(std::memory_order_relaxed);
            if ((active & kShapes) == 0) {
                return;
            }
            kernel_ = kernel;
            tensors_.reserve(inputs.size() + outputs.size());
            if ((active & kAttributes) != 0) {
                std::ostringstream attributes;
                attributes.imbue(std::locale::classic());
                attributes << std::boolalpha << std::setprecision(std::numeric_limits<float>::max_digits10);
                for (const auto& input : inputs) {
                    const size_t slot = tensors_.size();
                    if (AppendTensor(input.tensor, MSPROF_GE_TENSOR_TYPE_INPUT) && input.name) {
                        attributes << ' ' << input.name << "=Input[" << slot << ']';
                        if (input.loweredName) {
                            attributes << '(' << input.loweredName << ')';
                        }
                    }
                }
                // Each field written by the adapter starts with a space.
                write(attributes);
                attributes_ = attributes.str();
            } else {
                for (const auto& input : inputs) {
                    AppendTensor(input.tensor, MSPROF_GE_TENSOR_TYPE_INPUT);
                }
            }
            Append(outputs, MSPROF_GE_TENSOR_TYPE_OUTPUT);
        }

        uint64_t Begin() const
        {
            return !tensors_.empty() && IsEnabled() ? MsprofSysCycleTime() : 0;
        }

        void Report(uint64_t begin) const
        {
            if (begin == 0) {
                return;
            }
            const uint64_t end = MsprofSysCycleTime();
            // This item ID tells the exporter to keep the name from the runtime task.
            constexpr char kernel_name_ref[] = "Refer_to_kernelName";
            const uint64_t item_id = MsprofGetHashId(kernel_name_ref, sizeof(kernel_name_ref) - 1);
            const auto tid = static_cast<uint32_t>(syscall(SYS_gettid));
            MsprofApi launch{};
            launch.level = MSPROF_REPORT_NODE_LEVEL;
            launch.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
            launch.threadId = tid;
            launch.beginTime = begin;
            launch.endTime = end;
            launch.itemId = item_id;
            if (MsprofReportApi(1, &launch) != 0) {
                return;
            }

            MsprofAdditionalInfo event{};
            event.level = launch.level;
            event.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
            event.threadId = tid;
            event.timeStamp = begin + 1; // Match CANN's timestamp convention for additional records.
            static_assert(sizeof(MsprofTensorInfo) <= sizeof(event.data));
            event.dataLen = sizeof(MsprofTensorInfo);
            for (size_t offset = 0; offset < tensors_.size(); offset += MSPROF_GE_TENSOR_DATA_NUM) {
                MsprofTensorInfo info{};
                info.opName = item_id;
                info.tensorNum = std::min<size_t>(tensors_.size() - offset, MSPROF_GE_TENSOR_DATA_NUM);
                std::copy_n(tensors_.data() + offset, info.tensorNum, info.tensorData);
                std::memcpy(event.data, &info, sizeof(info));
                (void)MsprofReportAdditionalInfo(1, &event, sizeof(event));
            }
            if (kernel_) {
                ReportAttributes(kernel_, begin, attributes_);
            }
        }

        template <typename Launch>
        void operator()(Launch&& launch) const
        {
            const uint64_t begin = Begin();
            launch();
            Report(begin);
        }

      private:
        bool AppendTensor(const at::Tensor* tensor, uint32_t direction)
        {
            if (!tensor || !tensor->defined() || tensor->device().type() != at::kPrivateUse1 ||
                tensor->dim() > MSPROF_GE_TENSOR_DATA_SHAPE_LEN) {
                return false;
            }
            const auto sizes = tensor->sizes();
            if (std::any_of(sizes.begin(), sizes.end(),
                            [](int64_t size) { return size > std::numeric_limits<uint32_t>::max(); })) {
                return false;
            }
            auto& desc = tensors_.emplace_back();
            desc.tensorType = direction;
            desc.format = static_cast<uint32_t>(at_npu::native::get_npu_format(*tensor));
            desc.dataType = static_cast<uint32_t>(ProfileDataType(tensor->scalar_type()));
            std::copy(sizes.begin(), sizes.end(), desc.shape);
            return true;
        }

        void Append(std::initializer_list<const at::Tensor*> tensors, uint32_t direction)
        {
            for (const at::Tensor* tensor : tensors) {
                AppendTensor(tensor, direction);
            }
        }

        const char* kernel_ = nullptr;
        std::string attributes_;
        // OpCommand can launch on a different Host thread. Keep descriptors by value.
        std::vector<MsrofTensorData> tensors_;
    };

    inline std::shared_ptr<const TensorInfo> ShareTensors(std::initializer_list<const at::Tensor*> inputs,
                                                          std::initializer_list<const at::Tensor*> outputs)
    {
        return IsEnabled() ? std::make_shared<const TensorInfo>(inputs, outputs) : nullptr;
    }

    template <typename Launch>
    void LaunchWithProfile(const std::shared_ptr<const TensorInfo>& profile, Launch&& launch)
    {
        if (profile) {
            (*profile)(launch);
        } else {
            launch();
        }
    }

} // namespace fa_profile
