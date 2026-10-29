import shutil
from enum import Enum

import torch
from tabulate import tabulate

from uniattention.testing.utils import _filter_none_pairs
from uniattention.utils.device import get_torch_device, get_torch_device_module


# STYLE.SHOW will printout each benchmark of different operators
# STYLE.COMPARE will printout the benchmark comparison of  different operators under each peice of data
class STYLE(Enum):
    SHOW = 1
    COMPARE = 2


def print_benchmark(benchmark_dict: dict, output_file, style: STYLE = STYLE.COMPARE):
    def dual_print(*args, **kwargs):
        print(*args, **kwargs)
        if file_obj:
            print(*args, file=file_obj, **kwargs)

    file_obj = None

    if style == STYLE.SHOW:
        try:
            if output_file:
                file_obj = open(output_file, "w")

            terminal_width = shutil.get_terminal_size().columns
            dual_print("=" * terminal_width)
            dual_print("Configuration:")
            for key, value in benchmark_dict["config"].items():
                dual_print(f"  {key}: {value}")

            dual_print("*" * terminal_width)

            for func_name, metrics in benchmark_dict["func"].items():
                dual_print(f"\nFunction: {func_name}")

                headers = ["Data Point"]
                for metric, unit in benchmark_dict["benchmark"].items():
                    if unit != "N/A":
                        headers.append(f"{metric} ({unit})")
                    else:
                        headers.append(f"{metric}")

                rows = []
                num_data_points = len(metrics.get(next(iter(benchmark_dict["benchmark"].keys())), []))

                for i in range(num_data_points):
                    row = [i]
                    for metric in benchmark_dict["benchmark"].keys():
                        if metric in metrics and i < len(metrics[metric]):
                            row.append(metrics[metric][i])
                        else:
                            row.append("N/A")
                    rows.append(row)

                dual_print(tabulate(rows, headers=headers, tablefmt="grid"))
                dual_print("=" * terminal_width)

        finally:
            if file_obj:
                file_obj.close()

    elif style == STYLE.COMPARE:
        try:
            if output_file:
                file_obj = open(output_file, "w")

            terminal_width = shutil.get_terminal_size().columns
            dual_print("=" * terminal_width)
            dual_print("Configuration:")
            for key, value in benchmark_dict["config"].items():
                dual_print(f"  {key}: {value}")

            dual_print("*" * terminal_width)

            func_names = list(benchmark_dict["func"].keys())
            metrics = list(benchmark_dict["benchmark"].keys())

            first_func = func_names[0]
            first_metric = metrics[0]
            num_data_points = len(benchmark_dict["func"][first_func].get(first_metric, []))

            for i in range(num_data_points):
                dual_print(f"\nData Point: {i}")

                headers = ["Function"]
                for metric, unit in benchmark_dict["benchmark"].items():
                    if unit != "N/A":
                        headers.append(f"{metric} ({unit})")
                    else:
                        headers.append(f"{metric}")

                rows = []
                for func_name in func_names:
                    row = [func_name]
                    for metric in metrics:
                        if metric in benchmark_dict["func"][func_name] and i < len(
                            benchmark_dict["func"][func_name][metric]
                        ):
                            value = benchmark_dict["func"][func_name][metric][i]
                            if isinstance(value, float):
                                row.append(f"{value:.4f}")
                            else:
                                row.append(value)
                        else:
                            row.append("N/A")
                    rows.append(row)

                dual_print(tabulate(rows, headers=headers, tablefmt="grid"))

                dual_print("=" * terminal_width)
        finally:
            if file_obj:
                file_obj.close()


def do_ops_warmup(
    func,
    inputs: dict,
    warmup: int = 5,
    test_backward: bool = True,
):
    inputs_copy = {k: v.clone().detach() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    for _ in range(warmup):
        if test_backward:
            for tensor in inputs_copy.values():
                if isinstance(tensor, torch.Tensor) and torch.is_floating_point(tensor):
                    tensor.requires_grad = True

        # Forward
        ref_output = func(**inputs_copy)
        if isinstance(ref_output, torch.Tensor):
            ref_output = (ref_output,)

        # Backward
        if test_backward:
            dout = [
                torch.randn_like(ref_o) if isinstance(ref_o, torch.Tensor) and torch.is_floating_point(ref_o) else None
                for ref_o in ref_output
            ]

            filtered_dout, filtered_ref_output = _filter_none_pairs(*zip(dout, ref_output, strict=False))

            torch.autograd.backward(tensors=filtered_ref_output, grad_tensors=filtered_dout)


def do_ops_bench(
    func,
    inputs: dict,
    warmup: int = 5,
    rep: int = 50,
    test_backward: bool = True,
    pre_hook=None,
):
    event_record = {
        "fwd_elapsed_time": [],
        "bwd_elapsed_time": [],
    }
    torch_device = get_torch_device_module()

    # allocating 256MB to match L2 cache size
    x = torch.empty(int(256 * (1024**2)), dtype=torch.int8, device=get_torch_device())

    def flush_l2_cache():
        x.zero_()

    # warmup
    do_ops_warmup(func, inputs, warmup, test_backward)

    for r in range(rep):
        if test_backward:
            for tensor in inputs.values():
                if isinstance(tensor, torch.Tensor) and torch.is_floating_point(tensor):
                    tensor.requires_grad = True

        if pre_hook:
            pre_hook()

        flush_l2_cache()

        fwd_start_event = torch_device.Event(enable_timing=True)
        fwd_end_event = torch_device.Event(enable_timing=True)

        torch_device.synchronize()
        fwd_start_event.record()
        ref_output = func(**inputs)
        fwd_end_event.record()
        torch_device.synchronize()

        fwd_elapsed_time = fwd_start_event.elapsed_time(fwd_end_event)
        event_record["fwd_elapsed_time"].append(fwd_elapsed_time)

        if test_backward:
            if isinstance(ref_output, torch.Tensor):
                ref_output = (ref_output,)

            for tensor in inputs.values():
                if isinstance(tensor, torch.Tensor):
                    tensor.grad = None

            dout = [
                torch.randn_like(ref_o) if isinstance(ref_o, torch.Tensor) and torch.is_floating_point(ref_o) else None
                for ref_o in ref_output
            ]
            filtered_dout, filtered_ref_output = _filter_none_pairs(*zip(dout, ref_output, strict=False))
            bwd_start_event = torch_device.Event(enable_timing=True)
            bwd_end_event = torch_device.Event(enable_timing=True)

            torch_device.synchronize()
            bwd_start_event.record()
            torch.autograd.backward(tensors=filtered_ref_output, grad_tensors=filtered_dout)
            bwd_end_event.record()
            torch_device.synchronize()

            bwd_elapsed_time = bwd_start_event.elapsed_time(bwd_end_event)
            event_record["bwd_elapsed_time"].append(bwd_elapsed_time)

    return event_record


def ops_benchmark(
    test_func: dict,
    ref_func,
    inputs: dict,
    config: dict,
    benchmark_record: dict = None,
    warmup: int = 5,
    rep: int = 50,
    op_flops: float = 0.0,
    test_backward: bool = True,
    backward_flops_multiple: float = 2.0,
    pre_hook=None,
):
    if benchmark_record is None:
        benchmark_record = {
            "config": {"rep": rep, "warmup": warmup},
            "benchmark": {
                "fwd_elapsed_time": "ms",
                "fwd_tflops": "",
                "bwd_elapsed_time": "ms",
                "bwd_tflops": "",
                "fwd_bwd_total_time": "ms",
                "fwd_bwd_tflops": "",
            },
            "func": {},
        }
        for k in config.keys():
            benchmark_record["benchmark"].update({k: "N/A"})

    event_record = do_ops_bench(
        ref_func,
        inputs,
        warmup=warmup,
        rep=rep,
        test_backward=test_backward,
        pre_hook=pre_hook,
    )

    if "ref" not in benchmark_record["func"]:
        benchmark_record["func"]["ref"] = {
            "fwd_elapsed_time": [],
            "fwd_tflops": [],
            "bwd_elapsed_time": [],
            "bwd_tflops": [],
            "fwd_bwd_total_time": [],
            "fwd_bwd_tflops": [],
        }
        for k in config.keys():
            benchmark_record["func"]["ref"][k] = []

    fwd_t = sum(event_record["fwd_elapsed_time"]) / rep
    benchmark_record["func"]["ref"]["fwd_elapsed_time"].append(fwd_t)
    benchmark_record["func"]["ref"]["fwd_tflops"].append(op_flops / fwd_t * 1e-9)
    bwd_t = sum(event_record["bwd_elapsed_time"]) / rep
    benchmark_record["func"]["ref"]["bwd_elapsed_time"].append(bwd_t)
    benchmark_record["func"]["ref"]["bwd_tflops"].append(op_flops * backward_flops_multiple / bwd_t * 1e-9)
    benchmark_record["func"]["ref"]["fwd_bwd_total_time"].append(fwd_t + bwd_t)
    benchmark_record["func"]["ref"]["fwd_bwd_tflops"].append(
        op_flops * (1 + backward_flops_multiple) / (fwd_t + bwd_t) * 1e-9
    )

    for k in config.keys():
        benchmark_record["func"]["ref"][k].append(config[k])

    for name, func in test_func.items():
        func_event_record = do_ops_bench(
            func,
            inputs,
            warmup=warmup,
            rep=rep,
            test_backward=test_backward,
            pre_hook=pre_hook,
        )

        if name not in benchmark_record["func"]:
            benchmark_record["func"][name] = {
                "fwd_elapsed_time": [],
                "fwd_tflops": [],
                "bwd_elapsed_time": [],
                "bwd_tflops": [],
                "fwd_bwd_total_time": [],
                "fwd_bwd_tflops": [],
            }

            for k in config.keys():
                benchmark_record["func"][name][k] = []
        fwd_t = sum(func_event_record["fwd_elapsed_time"]) / rep
        benchmark_record["func"][name]["fwd_elapsed_time"].append(fwd_t)
        benchmark_record["func"][name]["fwd_tflops"].append(op_flops / fwd_t * 1e-9)
        bwd_t = sum(func_event_record["bwd_elapsed_time"]) / rep
        benchmark_record["func"][name]["bwd_elapsed_time"].append(bwd_t)
        benchmark_record["func"][name]["bwd_tflops"].append(op_flops * backward_flops_multiple / bwd_t * 1e-9)
        benchmark_record["func"][name]["fwd_bwd_total_time"].append(fwd_t + bwd_t)
        benchmark_record["func"][name]["fwd_bwd_tflops"].append(
            op_flops * (1 + backward_flops_multiple) / (fwd_t + bwd_t) * 1e-9
        )

        for k in config.keys():
            benchmark_record["func"][name][k].append(config[k])

    return benchmark_record


def ops_benchmark_with_different_inputs(
    test_func: dict,
    inputs: list,
    config: dict,
    benchmark_record: dict = None,
    warmup: int = 5,
    rep: int = 50,
    test_backward: bool = True,
    pre_hook=None,
):
    if benchmark_record is None:
        benchmark_record = {
            "config": {"rep": rep, "warmup": warmup},
            "benchmark": {
                "fwd_elapsed_time": "ms",
                "bwd_elapsed_time": "ms",
                "fwd_bwd_total_time": "ms",
            },
            "func": {},
        }
        for k in config.keys():
            benchmark_record["benchmark"][k] = "N/A"

    for index, (name, func) in enumerate(test_func.items()):
        func_event_record = do_ops_bench(
            func,
            inputs[index],
            warmup=warmup,
            rep=rep,
            test_backward=test_backward,
            pre_hook=pre_hook,
        )

        if name not in benchmark_record["func"]:
            benchmark_record["func"][name] = {"fwd_elapsed_time": [], "bwd_elapsed_time": [], "fwd_bwd_total_time": []}

            for k in config.keys():
                benchmark_record["func"][name][k] = []
        benchmark_record["func"][name]["fwd_elapsed_time"].append(sum(func_event_record["fwd_elapsed_time"]) / rep)
        benchmark_record["func"][name]["bwd_elapsed_time"].append(sum(func_event_record["bwd_elapsed_time"]) / rep)
        benchmark_record["func"][name]["fwd_bwd_total_time"].append(
            sum(func_event_record["fwd_elapsed_time"]) / rep + sum(func_event_record["bwd_elapsed_time"]) / rep
        )
        for k in config.keys():
            benchmark_record["func"][name][k].append(config[k])

    return benchmark_record
