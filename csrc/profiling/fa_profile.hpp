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

// Some torch_npu releases include their bundled ACL header first, hiding the
// CANN header's tensor definitions and, in older releases, the range declarations.
// Use the CANN header's documented guard and frozen version-1 ABI.
#ifndef ACL_PROF_TENSOR_INFO_DEFINED
#define ACL_PROF_TENSOR_INFO_DEFINED
#define ACL_PROF_EVENT_ATTR_VERSION 1
struct aclprofTensor {
    uint32_t type;
    uint32_t format;
    uint32_t dataType;
    uint32_t shapeDim;
    uint32_t shape[8];
};
struct aclprofTensorInfo {
    uint64_t opNameId;
    uint64_t opTypeId;
    uint32_t resv;
    uint32_t tensorNum;
    uint32_t kernelType;
    uint32_t blockNums;
    void* stream;
    aclprofTensor* tensors;
};
struct aclprofEventAttributes {
    uint16_t version;
    uint16_t size;
    uint32_t messageType;
    union Message {
        aclprofTensorInfo* tensorInfo;
    } message;
};

extern "C" {
MSVP_PROF_API uint64_t aclprofStr2Id(const char* message);
MSVP_PROF_API aclError aclprofRangePushEx(aclprofEventAttributes* attr);
MSVP_PROF_API aclError aclprofRangePop();
}
#endif

namespace fa_profile __attribute__((visibility("hidden"))) {

    constexpr uint32_t kTensorInfoMessage = 0;
    constexpr unsigned kShapes = 1;
    constexpr unsigned kAttributes = 2;
    constexpr unsigned kTaskL0TensorEvents = 4;
    inline std::atomic<unsigned> mode{0};
    inline std::atomic<uint64_t> sequence{0};

    inline bool SampleOptionEquals(std::string_view options, std::string_view key, std::string_view expected)
    {
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
        return value != std::string_view::npos && options.substr(value, expected.size()) == expected;
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
            unsigned flags = shapes ? kShapes : 0U;
            if (shapes) {
                const char* config = std::getenv("PROFILER_SAMPLECONFIG");
                const std::string_view options(config ? config : "");
                if ((command.profSwitch & PROF_MSPROFTX_MASK) != 0 ||
                    SampleOptionEquals(options, "\"msproftx\"", "\"on\"")) {
                    flags |= kAttributes;
                }
                // Keep software node correlation for in-process ACL sessions and for
                // the msprof task-time=l0 exporter configured to collect shapes.
                if (options.empty() || (SampleOptionEquals(options, "\"taskTime\"", "\"l0\"") &&
                                        SampleOptionEquals(options, "\"aicoreShape\"", "\"on\""))) {
                    flags |= kTaskL0TensorEvents;
                }
            }
            mode.store(flags, std::memory_order_relaxed);
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

        class LaunchRange {
          public:
            LaunchRange(const TensorInfo* profile, aclrtStream stream, uint32_t block_dim)
            {
                if (!profile || profile->tensors_.empty()) {
                    return;
                }
                const unsigned active = mode.load(std::memory_order_relaxed);
                if ((active & kShapes) == 0) {
                    return;
                }
                if ((active & kTaskL0TensorEvents) != 0) {
                    task_l0_profile_ = profile;
                    task_l0_begin_ = MsprofSysCycleTime();
                    return;
                }

                // Let CANN attach the descriptors to the node created by the
                // actual runtime launch on this stream and host thread.
                const uint64_t name = aclprofStr2Id("Refer_to_kernelName");
                if (name == std::numeric_limits<uint64_t>::max()) {
                    return;
                }
                info_ = {};
                info_.opNameId = name;
                info_.opTypeId = name;
                info_.tensorNum = static_cast<uint32_t>(profile->tensors_.size());
                info_.blockNums = block_dim;
                info_.stream = stream;
                // The API takes a mutable pointer but consumes this field as input.
                info_.tensors = const_cast<aclprofTensor*>(profile->tensors_.data());
                attributes_ = {};
                attributes_.version = ACL_PROF_EVENT_ATTR_VERSION;
                attributes_.size = sizeof(attributes_.message);
                attributes_.messageType = kTensorInfoMessage;
                attributes_.message.tensorInfo = &info_;
                pushed_ = aclprofRangePushEx(&attributes_) == ACL_SUCCESS;
            }

            LaunchRange(const LaunchRange&) = delete;
            LaunchRange& operator=(const LaunchRange&) = delete;

            ~LaunchRange()
            {
                Finish();
            }

            void Finish()
            {
                if (task_l0_begin_ != 0) {
                    task_l0_profile_->ReportTaskL0(task_l0_begin_);
                    task_l0_begin_ = 0;
                }
                if (pushed_) {
                    (void)aclprofRangePop();
                    pushed_ = false;
                }
            }

          private:
            aclprofTensorInfo info_;
            aclprofEventAttributes attributes_;
            const TensorInfo* task_l0_profile_ = nullptr;
            uint64_t task_l0_begin_ = 0;
            bool pushed_ = false;
        };

        template <typename Launch>
        void operator()(aclrtStream stream, uint32_t block_dim, Launch&& launch) const
        {
            const uint64_t begin = !attributes_.empty() && (mode.load(std::memory_order_relaxed) & kAttributes) != 0
                                       ? MsprofSysCycleTime()
                                       : 0;
            {
                LaunchRange range(this, stream, block_dim);
                launch();
            }
            if (begin != 0) {
                ReportAttributes(kernel_, begin, attributes_);
            }
        }

      private:
        void ReportTaskL0(uint64_t begin) const
        {
            const uint64_t end = MsprofSysCycleTime();
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
            event.timeStamp = begin + 1; // CANN's convention for additional software node records.
            static_assert(sizeof(MsprofTensorInfo) <= sizeof(event.data));
            event.dataLen = sizeof(MsprofTensorInfo);
            for (size_t offset = 0; offset < tensors_.size(); offset += MSPROF_GE_TENSOR_DATA_NUM) {
                MsprofTensorInfo batch{};
                batch.opName = item_id;
                batch.tensorNum = std::min<size_t>(tensors_.size() - offset, MSPROF_GE_TENSOR_DATA_NUM);
                for (uint32_t i = 0; i < batch.tensorNum; ++i) {
                    const auto& src = tensors_[offset + i];
                    auto& dst = batch.tensorData[i];
                    dst.tensorType = src.type;
                    dst.format = src.format;
                    dst.dataType = src.dataType;
                    std::copy_n(src.shape, src.shapeDim, dst.shape);
                }
                std::memcpy(event.data, &batch, sizeof(batch));
                (void)MsprofReportAdditionalInfo(1, &event, sizeof(event));
            }
        }

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
            desc.type = direction;
            desc.format = static_cast<uint32_t>(at_npu::native::get_npu_format(*tensor));
            desc.dataType = static_cast<uint32_t>(ProfileDataType(tensor->scalar_type()));
            desc.shapeDim = static_cast<uint32_t>(tensor->dim());
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
        std::vector<aclprofTensor> tensors_;
    };

    inline std::shared_ptr<const TensorInfo> ShareTensors(std::initializer_list<const at::Tensor*> inputs,
                                                          std::initializer_list<const at::Tensor*> outputs)
    {
        return IsEnabled() ? std::make_shared<const TensorInfo>(inputs, outputs) : nullptr;
    }

    template <typename Launch>
    void LaunchWithProfile(const std::shared_ptr<const TensorInfo>& profile, aclrtStream stream, uint32_t block_dim,
                           Launch&& launch)
    {
        if (profile) {
            (*profile)(stream, block_dim, launch);
        } else {
            launch();
        }
    }

} // namespace fa_profile
