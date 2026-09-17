#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

import os
import unittest
import logging
import json
import re
import numpy
import torch
import torch_npu
import math
import sys
import shutil
from enum import Enum

MIN_ERR = 1e-7
class OpTypes(Enum):
    NA = 0 # new standard is not available
    MOVE = 1
    RAND = 2
    CAST = 3
    COMPUTE_INTEGER = 4
    COMPUTE_QUANT = 5
    COMPUTE_FLOAT = 6
    COMPUTE_FLOAT_HIGH_PRECISION = 7
    VECTOR_FUSION = 8
    CV_FUSION = 9

dtype_dict = {"float": torch.float32, "float16": torch.float16, "int8": torch.int8, "int32": torch.int32, "uint8": torch.uint8,
              "int16": torch.int16, "uint16": torch.int16, "uint32": torch.int32, "int64": torch.int64, "uint64": torch.int64,
              "double": torch.double, "bool": torch.bool, "complex64": torch.complex64, "complex128": torch.complex128, "bf16": torch.bfloat16}

def get_eb_threshold(dtype:torch.dtype):
    eb_threshold = 0
    if dtype in [torch.bfloat16]:
        eb_threshold = 2**(-7)
    if dtype in [torch.float16]:
        eb_threshold = 2**(-10)
    if dtype in [torch.float32]:
        eb_threshold = 2**(-14)
    return eb_threshold

def get_err_threshold(op_type:OpTypes, dtype:torch.dtype):
    err_threshold = 0
    if op_type in [OpTypes.MOVE, OpTypes.RAND, OpTypes.CAST, OpTypes.COMPUTE_INTEGER]:
        pass
    if op_type in [OpTypes.COMPUTE_QUANT, OpTypes.COMPUTE_FLOAT]:
        if dtype in [torch.bfloat16]:
            err_threshold = 2**(-7)
        if dtype in [torch.float16]:
            err_threshold = 2**(-8)
        if dtype in [torch.float32]:
            err_threshold = 2**(-11)
    if op_type in [OpTypes.CV_FUSION]:
        if dtype in [torch.bfloat16]:
            err_threshold = 2**(-8)
        if dtype in [torch.float16]:
            err_threshold = 2**(-11)
        if dtype in [torch.float32]:
            err_threshold = 2**(-14)
    return err_threshold


#误差均衡性（EB）
def get_eb(golden:torch.Tensor, actual:torch.Tensor):
    golden = golden.to(torch.float32)
    golden_nmax = torch.clamp(torch.abs(golden), min = 1)
    actual_error = actual.to(torch.float32) - golden
    EB = torch.mean(actual_error / golden_nmax)
    return EB

#单标杆、浮点比对方法|actual - expected| <= err × max(1, | expected |)
def ref_compare(golden:torch.Tensor, actual:torch.Tensor, err):
    golden = golden.to(torch.float32)
    golden_nmax = torch.clamp(torch.abs(golden), min = 1)
    abs_error = torch.abs(actual.to(torch.float32) - golden)
    result = (abs_error <= err * golden_nmax).all()
    logging.debug(f"new golden result:{result}")
    return result


#最大相对误差：max relative error，MARE
def get_mare(golden:torch.Tensor, actual:torch.Tensor):
    golden = golden.to(torch.float32)
    abs_error = torch.abs(actual.to(torch.float32) - golden) / (torch.abs(golden) + MIN_ERR)
    mare = torch.max(abs_error.flatten())
    return mare

#平均相对误差：mean relative error，MERE
def get_mere(golden:torch.Tensor, actual:torch.Tensor):
    golden = golden.to(torch.float32)
    abs_error = torch.abs(actual.to(torch.float32) - golden) / (torch.abs(golden) + MIN_ERR)
    mere = torch.mean(abs_error)
    return mere

#均方根误差:Root Mean Squared Error，RMSE
def get_rmse(golden:torch.Tensor, actual:torch.Tensor):
    golden = golden.to(torch.float32)
    sqr_err = torch.pow((actual.to(torch.float32) - golden), 2)
    rmse = torch.sqrt(torch.mean(sqr_err))
    return rmse

def compare_cv(golden:torch.Tensor, actual:torch.Tensor):
    op_type = OpTypes.COMPUTE_FLOAT
    eb_threshold = get_eb_threshold(actual.dtype)
    err_threshold = get_err_threshold(op_type, actual.dtype)
    print(f"err_threshold:{err_threshold} eb_threshold:{eb_threshold}")

    mare_npu = get_mare(golden, actual)
    mere_npu = get_mere(golden, actual)
    rmse_npu = get_rmse(golden, actual)
    eb_npu = get_eb(golden, actual)

    # 多维度联合判决：MARE 放宽至 15×err_thd（适配多算子链路），其余维保持 1×err_thd
    result = (mare_npu < err_threshold * 15) and (mere_npu < err_threshold) and (rmse_npu < err_threshold) and (abs(eb_npu) < eb_threshold)

    print(f"mare_npu:{mare_npu} mere_npu:{mere_npu} rmse_npu:{rmse_npu} EB:{eb_npu}")
    print(f"precision check result:{result}")
    return result


def compare_rule(golden: torch.Tensor, actual: torch.Tensor,
                 ratios=(0.001, 0.001, 0.005, 0.005)):
    """
    逐元素精度对比，与 gen_ours_output.py 的 compare_rule 逻辑一致。

    ratios: (rel_limit, abs_floor, strict_rel_limit, strict_abs_floor)
        - limit_error = max(|golden| * ratios[0], ratios[1])
        - strict_limit_error = max(|golden| * ratios[2], ratios[3])
        - 判决通过条件: strict_error_count / total <= ratios[2] (即 > 99.5% 元素在 5‰ 误差内)

    返回: (max_diff, result_bool)
    """
    golden = golden.flatten().to(torch.float32)
    actual = actual.flatten().to(torch.float32)
    total = actual.shape[0]

    diff = torch.abs(golden - actual)
    max_diff = diff.max().item()

    limit_error = torch.maximum(torch.abs(golden * ratios[0]), torch.tensor(ratios[1]))
    strict_limit_error = torch.maximum(torch.abs(golden * ratios[2]), torch.tensor(ratios[3]))

    error_count = torch.gt(diff, limit_error).sum().item()
    strict_error_count = torch.gt(diff, strict_limit_error).sum().item()

    result = (float(strict_error_count) / total) <= ratios[2]
    if not result:
        print(f"maxDiff {max_diff:.6e}")
        print(f"1/1000 Accuracy is {1 - float(error_count) / total:.6f}")
        print(f"5/1000 Accuracy is {1 - float(strict_error_count) / total:.6f}")
        print("compare failed&")
    return max_diff, result