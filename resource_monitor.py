import argparse
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


CPU_FIELDS = (
    "user",
    "nice",
    "system",
    "idle",
    "iowait",
    "irq",
    "softirq",
    "steal",
    "guest",
    "guest_nice",
)


def read_proc_stat(path="/proc/stat"):
    with Path(path).open("r", encoding="utf-8") as handle:
        fields = handle.readline().split()
    if not fields or fields[0] != "cpu":
        raise ValueError(f"Could not parse CPU totals from {path}")
    values = [int(value) for value in fields[1:]]
    totals = {
        name: values[idx] if idx < len(values) else 0
        for idx, name in enumerate(CPU_FIELDS)
    }
    totals["idle_raw"] = totals["idle"]
    totals["idle_all"] = totals["idle"] + totals["iowait"]
    totals["idle"] = totals["idle_all"]
    totals["total"] = sum(values)
    return totals


def cpu_percent_between(previous, current):
    total_delta = current["total"] - previous["total"]
    idle_delta = current["idle"] - previous["idle"]
    if total_delta <= 0:
        return None
    busy = max(0, total_delta - max(0, idle_delta))
    return 100.0 * busy / total_delta


def cpu_percent_breakdown_between(previous, current):
    total_delta = current["total"] - previous["total"]
    if total_delta <= 0:
        return {}
    breakdown = {}
    for field in CPU_FIELDS:
        delta = max(0, current.get(field, 0) - previous.get(field, 0))
        breakdown[f"{field}_percent"] = 100.0 * delta / total_delta
    busy_percent = cpu_percent_between(previous, current)
    if busy_percent is not None:
        breakdown["busy_percent"] = busy_percent
    return breakdown


def parse_cpuset_count(text):
    count = 0
    for chunk in str(text).strip().split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid cpuset range: {chunk!r}")
            count += end - start + 1
        else:
            int(chunk)
            count += 1
    return count


def read_cpuset_count(path="/sys/fs/cgroup/cpuset.cpus.effective"):
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        return parse_cpuset_count(text)
    except ValueError:
        return None


def read_cgroup_cpu_max(path="/sys/fs/cgroup/cpu.max"):
    try:
        parts = Path(path).read_text(encoding="utf-8").split()
    except OSError:
        return {}
    if not parts:
        return {}
    period_usec = int(parts[1]) if len(parts) > 1 else 100000
    if parts[0] == "max":
        return {
            "quota_usec": None,
            "period_usec": period_usec,
            "quota_cores": None,
            "unlimited": True,
        }
    quota_usec = int(parts[0])
    quota_cores = quota_usec / period_usec if period_usec > 0 else None
    return {
        "quota_usec": quota_usec,
        "period_usec": period_usec,
        "quota_cores": quota_cores,
        "unlimited": False,
    }


def read_cgroup_cpu_stat(path="/sys/fs/cgroup/cpu.stat"):
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    values = {}
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            values[parts[0]] = int(parts[1])
        except ValueError:
            continue
    return values


def cpu_capacity_cores(cpu_max=None, cpuset_count=None, visible_cpus=None):
    visible = cpuset_count or visible_cpus or os.cpu_count() or 1
    capacity = float(visible)
    quota_cores = (cpu_max or {}).get("quota_cores")
    if isinstance(quota_cores, (int, float)) and quota_cores > 0:
        capacity = min(capacity, float(quota_cores))
    return max(1.0, capacity)


def usable_cpu_count(multiplier=1.0):
    capacity = cpu_capacity_cores(
        cpu_max=read_cgroup_cpu_max(),
        cpuset_count=read_cpuset_count(),
        visible_cpus=os.cpu_count(),
    )
    return max(1, int(math.ceil(capacity * max(0.0, float(multiplier or 1.0)))))


def cgroup_cpu_metrics_between(previous, current, elapsed_s, capacity_cores):
    if not previous or not current or elapsed_s <= 0:
        return {}
    usage_delta_usec = max(
        0,
        current.get("usage_usec", 0) - previous.get("usage_usec", 0),
    )
    used_cores = usage_delta_usec / 1_000_000.0 / elapsed_s
    metrics = {
        "cpu_usage_usec_delta": usage_delta_usec,
        "cpu_used_cores": used_cores,
    }
    if capacity_cores and capacity_cores > 0:
        metrics["cpu_util_percent"] = 100.0 * used_cores / capacity_cores

    throttled_usec_delta = max(
        0,
        current.get("throttled_usec", 0) - previous.get("throttled_usec", 0),
    )
    throttled_periods_delta = max(
        0,
        current.get("nr_throttled", 0) - previous.get("nr_throttled", 0),
    )
    periods_delta = max(
        0,
        current.get("nr_periods", 0) - previous.get("nr_periods", 0),
    )
    metrics["cpu_throttled_usec_delta"] = throttled_usec_delta
    metrics["cpu_throttled_periods_delta"] = throttled_periods_delta
    metrics["cpu_periods_delta"] = periods_delta
    if periods_delta > 0:
        metrics["cpu_throttled_period_fraction"] = (
            throttled_periods_delta / periods_delta
        )
    return metrics


def resource_tensorboard_scalars(sample):
    scalars = {}

    def add(tag, value):
        if isinstance(value, (int, float)):
            scalars[tag] = float(value)

    add("resource/cpu_util_percent", sample.get("cpu_util_percent"))
    add("resource/cpu_used_cores", sample.get("cpu_used_cores"))
    add("resource/host_cpu_util_percent", sample.get("host_cpu_util_percent"))
    add(
        "resource/cpu_throttled_period_fraction",
        sample.get("cpu_throttled_period_fraction"),
    )
    cpu = sample.get("cpu") or {}
    add("resource/cpu_capacity_cores", cpu.get("capacity_cores"))
    add("resource/cpu_quota_cores", cpu.get("quota_cores"))
    add("resource/cpu_usable_workers", cpu.get("usable_workers"))

    memory = sample.get("memory") or {}
    add("resource/memory_used_percent", memory.get("memory_used_percent"))
    add("resource/memory_used_mb", memory.get("memory_used_mb"))

    gpu = sample.get("gpu") or {}
    add("resource/gpu_util_percent", gpu.get("gpu_util_percent"))
    add("resource/gpu_memory_used_mb", gpu.get("gpu_memory_used_mb"))
    add("resource/gpu_temperature_c", gpu.get("gpu_temperature_c"))
    return scalars


def tensorboard_writer(logdir):
    if not logdir:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        return None
    return SummaryWriter(str(logdir))


def read_loadavg(path="/proc/loadavg"):
    text = Path(path).read_text(encoding="utf-8").split()
    return {
        "load1": float(text[0]),
        "load5": float(text[1]),
        "load15": float(text[2]),
    }


def read_meminfo(path="/proc/meminfo"):
    values = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            key, raw_value = line.split(":", 1)
            parts = raw_value.strip().split()
            if parts and parts[0].isdigit():
                values[key] = int(parts[0])
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if total is None or available is None:
        return {}
    used = total - available
    return {
        "memory_total_mb": total / 1024,
        "memory_available_mb": available / 1024,
        "memory_used_mb": used / 1024,
        "memory_used_percent": 100.0 * used / total,
    }


def parse_nvidia_smi_query(output):
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return None
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) < 4:
        raise ValueError(f"Unexpected nvidia-smi query output: {lines[0]!r}")
    return {
        "gpu_util_percent": int(fields[0]),
        "gpu_memory_used_mb": int(fields[1]),
        "gpu_memory_total_mb": int(fields[2]),
        "gpu_temperature_c": int(fields[3]),
    }


def query_nvidia_smi(timeout=5):
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"gpu_error": str(exc)}
    if result.returncode != 0:
        return {"gpu_error": result.stderr.strip() or result.stdout.strip()}
    try:
        return parse_nvidia_smi_query(result.stdout) or {}
    except ValueError as exc:
        return {"gpu_error": str(exc)}


def read_resource_state():
    return {
        "monotonic_s": time.monotonic(),
        "host_cpu": read_proc_stat(),
        "cgroup_cpu": read_cgroup_cpu_stat(),
        "cpu_max": read_cgroup_cpu_max(),
        "cpuset_count": read_cpuset_count(),
        "visible_cpus": os.cpu_count(),
    }


def sample_resources(previous_state=None, include_gpu=True):
    current_state = read_resource_state()
    cpu_capacity = cpu_capacity_cores(
        cpu_max=current_state.get("cpu_max"),
        cpuset_count=current_state.get("cpuset_count"),
        visible_cpus=current_state.get("visible_cpus"),
    )
    sample = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "load": read_loadavg(),
        "memory": read_meminfo(),
        "cpu": {
            "capacity_cores": cpu_capacity,
            "quota_cores": current_state.get("cpu_max", {}).get("quota_cores"),
            "quota_usec": current_state.get("cpu_max", {}).get("quota_usec"),
            "period_usec": current_state.get("cpu_max", {}).get("period_usec"),
            "cpuset_cpus": current_state.get("cpuset_count"),
            "visible_cpus": current_state.get("visible_cpus"),
            "usable_workers": usable_cpu_count(),
        },
    }
    if previous_state is not None:
        previous_host_cpu = previous_state.get("host_cpu", previous_state)
        host_cpu = current_state["host_cpu"]
        host_cpu_util = cpu_percent_between(previous_host_cpu, host_cpu)
        if host_cpu_util is not None:
            sample["host_cpu_util_percent"] = host_cpu_util
            sample["host_cpu_breakdown_percent"] = cpu_percent_breakdown_between(
                previous_host_cpu,
                host_cpu,
            )

        elapsed_s = (
            current_state["monotonic_s"]
            - previous_state.get("monotonic_s", current_state["monotonic_s"])
        )
        cgroup_metrics = cgroup_cpu_metrics_between(
            previous_state.get("cgroup_cpu", {}),
            current_state.get("cgroup_cpu", {}),
            elapsed_s,
            cpu_capacity,
        )
        if cgroup_metrics:
            sample.update(cgroup_metrics)
        elif host_cpu_util is not None:
            sample["cpu_util_percent"] = host_cpu_util
    if include_gpu:
        sample["gpu"] = query_nvidia_smi()
    return sample, current_state


def monitor_resources(
    output_path,
    interval_s=60.0,
    samples=0,
    include_gpu=True,
    tensorboard_logdir=None,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    previous_state = read_resource_state()
    count = 0
    writer = tensorboard_writer(tensorboard_logdir)
    with output_path.open("a", encoding="utf-8") as handle:
        try:
            while samples <= 0 or count < samples:
                time.sleep(max(0.0, interval_s))
                sample, previous_state = sample_resources(
                    previous_state=previous_state,
                    include_gpu=include_gpu,
                )
                json.dump(sample, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                if writer is not None:
                    for tag, value in resource_tensorboard_scalars(sample).items():
                        writer.add_scalar(tag, value, count)
                    writer.flush()
                count += 1
        finally:
            if writer is not None:
                writer.close()


def main():
    parser = argparse.ArgumentParser(description="Write CPU/GPU resource samples as JSONL.")
    parser.add_argument(
        "--output",
        default="resource_monitor.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="Seconds between samples",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Number of samples to write; 0 means run until stopped",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Skip nvidia-smi polling",
    )
    parser.add_argument(
        "--tensorboard-logdir",
        default="checkpoints/tensorboard/resource_monitor",
        help="Optional TensorBoard logdir for CPU/GPU resource scalars",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard resource scalar mirroring",
    )
    args = parser.parse_args()
    monitor_resources(
        args.output,
        interval_s=args.interval,
        samples=args.samples,
        include_gpu=not args.no_gpu,
        tensorboard_logdir=None if args.no_tensorboard else args.tensorboard_logdir,
    )


if __name__ == "__main__":
    main()
