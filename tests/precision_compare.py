# Copyright (c) 2026, Minghua Shen.

import numpy as np
import datetime

def print_log(data=None):
    print("[♪] ", data)

def print_log_error(data=None, is_digit=True):
    if is_digit:
        print("[☢] ", data)
    else:
        print("[☢] ", "\033[31m%s\033[0m" % data)

def cal_relative_diff_np(real_data, expect_data, diff_thd):
    a = np.abs(np.subtract(real_data, expect_data))
    b1 = np.maximum(np.abs(real_data), (np.abs(expect_data)))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    result = np.where(a < diff_thd, a, a / b)
    return result

def cal_relative_diff(real_data, expect_data, diff_thd, type_str='fp16'):
    if 'nan' in str(expect_data) or 'inf' in str(expect_data):
        if type_str.lower() == 'fp16':
            expect_data = 65504
        else:
            expect_data = 3.4028e38
    diff = abs(float(real_data) - float(expect_data))
    if abs(float(real_data) - float(expect_data)) < diff_thd:
        result = diff
    else:
        result = diff / (float(max(abs(real_data), abs(expect_data))) + 10e-10)
    return result


def display_output(real_data, expect_data, start, end, diff_thd):
    def display_inner(idx):
        j = idx + start
        diff_rate = cal_relative_diff(expect_data[j], real_data[j], diff_thd)
        if "inf" in str(expect_data[j]) or "nan" in str(expect_data[j]):
            diff_abs = "inf" if "inf" in str(expect_data[j]) else "nan"
            print_log('%08d \t %-7s \t %-7s \t %-7s \t %-7s' % (start + idx, expect_data[j], real_data[j], diff_abs, diff_rate))
        else:
            diff_abs = abs(np.float64(expect_data[j]) - np.float64(real_data[j]))
            print_log('%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f' % (start + idx, expect_data[j], real_data[j], diff_abs, diff_rate))

    print_log('------------------------------------------------------------------------')
    print_log('Loop \t Expect(R) \t Real(L) \t L-R \t   \t |(L-R)/L|')
    print_log('------------------------------------------------------------------------')
    split_count = int(end - start)
    if split_count <= 100:
        for i in range(split_count + 1):
            display_inner(i)
    else:
        for i in range(10):
            display_inner(i)
        print_log('  ...  \t    ...   \t    ...   \t    ...    \t    ...')
        for i in range(split_count - 10 + 1, split_count + 1):
            display_inner(i)


def display_error_output(real_data, expect_data, err_idx, relative_diff):
    print("\033[31m------------------------------------error------------------------------------\033[0m")
    print_log_error('Loop \t Expect(R) \t Real(L) \t L-R \t   \t |(L-R)/L|')
    print_log_error('------------------------------------------------------------------------', is_digit=False)
    count = 0
    len_err = len(err_idx)
    for i in err_idx:
        count += 1
        if count < 10 or (90 < count < 100):
            print_log_error('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % 
                (i, expect_data[i], real_data[i], abs(np.float64(expect_data[i]) - np.float64(real_data[i])), relative_diff[count - 1]))
        elif count == 10 or (count == 100 and len_err > 100):
            dot_3 = '...'
            print_log_error('%05s\t %06s \t %06s \t %06s \t %06s' %(dot_3, dot_3, dot_3, dot_3, dot_3))
        elif count > 100:
            break

    print_log_error('----------------------------Max-RE-line---------------------------------', is_digit=False)
    max_error = max(relative_diff)
    m_idx_list = err_idx[np.where(relative_diff == max_error)]
    m_count = 0
    for m_idx in m_idx_list:
        m_count += 1
        if m_count < 4:
            print_log_error('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                m_idx, expect_data[m_idx], real_data[m_idx],
                abs(np.float64(expect_data[m_idx]) - np.float64(real_data[m_idx])), max_error))
        else:
            break
    print("\033[31m-----------------------------------------------------------------------------\033[0m")

# diff_thd：实际值减去预期值，误差在千分之一
# pct_thd： 通过数/用例总数，fp16 精度是99.90%
# 判断用例是否通过：首先去判断pct_thd是不是大于99.90%，如果大于，则Result pass，如果小于，则Result failed
# 如果pct_thd不是100%，则从所有用例中取出误差最大的值打印出来，这个值就是控制台的Maximum error is: 1.0. Tolerance threshold is: 0.1，
# 如果有值大于0.1，直接failed，如果小于0.1，则pass

# Loop：对比单点索引号
# Expect(R)：cpu期待输出值，即输入的右参数
# Real(L)：npu实际输出值，即输入的左参数
# L-R：单点绝对误差
# |(L-R)/L|： 单点相对误差
# DiffThd：相对误差容许阈值
# PctThd：预测正确点个数达标阈值
# PctRlt：相似度
# Result：预测结果
def data_compare(npu_output, cpu_output, diff_thd=0.005, pct_thd=0.005, max_diff_hd=0.1):
    if npu_output.dtype == "|V2":
        import bfloat16ext
        npu_output.dtype = "bfloat16"
    max_error_idx = 10000000
    real_data = npu_output.flatten()
    data_compe = cpu_output.flatten()
    if real_data.size == 0 and real_data.size == data_compe.size:
        print_log('The npu_output is [],and it is same as bm_output, the result of data_compare is \"Pass\"')
        return "Pass", 100.0, 0
    start = 0
    end = real_data.size - 1
    if end < start:
        end = start
    max_error = 0
    result = "Failed"
    if real_data.size != data_compe.size:
        print_log('Error,the size of npu output[%s] and benchmark[%s] is not equal.' % (real_data.size, data_compe.size))
        return result, 0.0, max_error

    overflows_count = data_compe[np.isinf(data_compe)].size + data_compe[np.isnan(data_compe)].size
    if overflows_count > 0:
        print_log('Overflow,size:%s,benchmark_output:%s, %s' % (overflows_count, data_compe[np.isinf(data_compe)][0:10], data_compe[np.isnan(data_compe)][0:10]))

    split_count = int(end - start + 1) if end != start else 1
    print_log('------------------------------------------------------------------------')
    print_log('该精度验证结果只能初步快速验证，不可依赖该结论！')
    print_log('------------------------------------------------------------------------')
    print_log('split_count:%s;  max_diff_hd:%s;' %(float(split_count), max_diff_hd))
    try:
        diff_abs = np.abs(np.subtract(real_data.astype(np.float32), data_compe.astype(np.float32)))
    except MemoryError:
        return result, 0.0, max_error
    diff_index = np.where(diff_abs > 0)
    rdiff = cal_relative_diff_np(real_data[diff_index].astype(np.float32), data_compe[diff_index].astype(np.float32), diff_thd)
    err_diff = rdiff[rdiff > diff_thd]
    diff_idx_list = diff_index[0]
    err_idx = diff_idx_list[np.where(rdiff > diff_thd)]
    fulfill_percent = float(split_count - err_diff.size) / float(split_count) * 100.0

    display_output(real_data, data_compe, start, end, diff_thd)
    pct_thd = (1 - pct_thd) * 100.0
    result = "success" if (fulfill_percent >= pct_thd) else "failed"
    if len(err_diff) > 0:
        max_error = max(err_diff[0:max_error_idx])
        if max_error >= max_diff_hd:
            result = "failed"
    print_log('------------------------------------------------------------------------')
    print_log('DiffThd  \t PctThd   \t PctRlt   \t Result')
    print_log('------------------------------------------------------------------------')
    print_log('%.4f  \t %.2f%%   \t %.6f%%   \t %s' %
              (diff_thd, pct_thd, fulfill_percent, result))
    print_log('------------------------------------------------------------------------')
    if len(err_diff) > 0:
        print_log('Max-RelativeError is: %s. Threshold is: %s.' %(max_error, max_diff_hd))
        print_log('------------------------------------------------------------------------')
    if result == "failed":
        display_error_output(real_data, data_compe, err_idx, err_diff[0:max_error_idx])
    return result, fulfill_percent, max_error