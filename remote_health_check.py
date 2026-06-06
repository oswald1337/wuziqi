import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from resource_monitor import query_nvidia_smi
from tensorboard_audit import (
    audit_resource_monitor_logdir,
    audit_tensorboard_logdir,
)
from training_audit import _parse_iso_timestamp


REQUIRED_TMUX_SESSIONS = ("tb_wuziqi", "gpu_monitor", "resource_monitor")


def parse_tmux_sessions(output):
    sessions = set()
    for line in output.splitlines():
        if ":" not in line:
            continue
        sessions.add(line.split(":", 1)[0].strip())
    return sessions


def tmux_sessions():
    result = subprocess.run(
        ["tmux", "ls"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return set(), result.stderr.strip() or result.stdout.strip()
    return parse_tmux_sessions(result.stdout), None


def check_url(url, timeout=3.0):
    try:
        with urlopen(url, timeout=timeout) as response:
            return {"ok": 200 <= response.status < 400, "status": response.status}
    except URLError as exc:
        return {"ok": False, "error": str(exc)}
    except TimeoutError as exc:
        return {"ok": False, "error": str(exc)}


def latest_jsonl_record(path):
    path = Path(path)
    if not path.exists():
        return None
    latest = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                latest = json.loads(line)
    return latest


def record_age_seconds(record, now=None):
    if not record:
        return None
    timestamp = _parse_iso_timestamp(record.get("timestamp"))
    if timestamp is None:
        return None
    now = now or datetime.now(timezone.utc)
    return max(0.0, (now - timestamp).total_seconds())


def check_resource_log(path, max_age_s=120.0):
    record = latest_jsonl_record(path)
    age_s = record_age_seconds(record)
    return {
        "ok": age_s is not None and age_s <= max_age_s,
        "path": str(path),
        "latest_timestamp": None if record is None else record.get("timestamp"),
        "age_s": age_s,
        "latest": record,
    }


def check_torch_cuda():
    try:
        import torch
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    result = {
        "ok": cuda_available and device_count > 0,
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": cuda_available,
        "device_count": device_count,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
    }
    if cuda_available and device_count > 0:
        try:
            result["device_name"] = torch.cuda.get_device_name(0)
        except Exception as exc:
            result["device_name_error"] = str(exc)
    return result


def check_parallel_readiness(preset="large_16x16_top_human_gpu", eval_games=32):
    try:
        from config import get_training_preset
        from train import _parallel_self_play_worker_count, _resolve_parallel_worker_count
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    try:
        config = get_training_preset(preset)
        self_play_workers = _parallel_self_play_worker_count(config)
        eval_workers = _resolve_parallel_worker_count(
            config,
            key="eval_parallel_games",
            prefix="eval",
            limit=eval_games,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc), "preset": preset}
    return {
        "ok": self_play_workers >= 1 and eval_workers >= 1,
        "preset": preset,
        "requested_self_play_parallel_games": config.get("self_play_parallel_games"),
        "resolved_self_play_workers": self_play_workers,
        "requested_eval_parallel_games": config.get("eval_parallel_games"),
        "resolved_eval_workers": eval_workers,
        "eval_games": eval_games,
        "gpu_inference_max_batch_size": config.get("gpu_inference_max_batch_size"),
        "gpu_inference_coalesce_ms": config.get("gpu_inference_coalesce_ms"),
        "gpu_inference_coalesce_slice_ms": config.get("gpu_inference_coalesce_slice_ms"),
        "gpu_inference_compact_request": config.get("gpu_inference_compact_request"),
        "gpu_inference_state_dtype": config.get("gpu_inference_state_dtype"),
        "gpu_inference_compact_response": config.get("gpu_inference_compact_response"),
        "mcts_batch_size": config.get("mcts_batch_size"),
        "mcts_min_batches_per_search": config.get("mcts_min_batches_per_search"),
    }


def health_warnings(
    checks,
    cpu_warn_percent=80.0,
    gpu_warn_percent=10.0,
    gpu_memory_warn_mb=256,
):
    warnings = []
    latest_resource = checks.get("resource_log", {}).get("latest") or {}
    cpu_util = latest_resource.get("cpu_util_percent")
    if isinstance(cpu_util, (int, float)) and cpu_util >= cpu_warn_percent:
        source = "container " if latest_resource.get("cpu") else ""
        warnings.append(
            f"latest resource monitor {source}CPU utilization is "
            f"{cpu_util:.1f}% before training; compare future run samples "
            "against this busy baseline"
        )
    host_cpu_util = latest_resource.get("host_cpu_util_percent")
    if (
        isinstance(host_cpu_util, (int, float))
        and host_cpu_util >= cpu_warn_percent
        and not (isinstance(cpu_util, (int, float)) and cpu_util >= cpu_warn_percent)
    ):
        warnings.append(
            "resource monitor sees high host-wide CPU utilization "
            f"{host_cpu_util:.1f}% before training; this can include other "
            "workloads outside the training container"
        )

    resource_gpu = latest_resource.get("gpu") or {}
    resource_gpu_util = resource_gpu.get("gpu_util_percent")
    if isinstance(resource_gpu_util, (int, float)) and resource_gpu_util >= gpu_warn_percent:
        warnings.append(
            "resource monitor sees non-idle GPU utilization before training: "
            f"{resource_gpu_util:.1f}%"
        )

    nvidia = checks.get("nvidia_smi") or {}
    gpu_util = nvidia.get("gpu_util_percent")
    if isinstance(gpu_util, (int, float)) and gpu_util >= gpu_warn_percent:
        warnings.append(
            f"nvidia-smi reports GPU utilization {gpu_util:.1f}% before training"
        )
    gpu_memory = nvidia.get("gpu_memory_used_mb")
    if isinstance(gpu_memory, (int, float)) and gpu_memory >= gpu_memory_warn_mb:
        warnings.append(
            f"nvidia-smi reports {gpu_memory:.0f}MiB GPU memory in use before training"
        )

    return warnings


def run_health_check(
    tensorboard_url="http://127.0.0.1:8081",
    tensorboard_logdir="checkpoints/tensorboard/large_16x16_top_human_gpu",
    resource_tensorboard_logdir="checkpoints/tensorboard/resource_monitor",
    resource_log="resource_monitor.jsonl",
    max_resource_age_s=120.0,
    required_tmux_sessions=REQUIRED_TMUX_SESSIONS,
    preset="large_16x16_top_human_gpu",
    eval_games=32,
):
    sessions, tmux_error = tmux_sessions()
    missing_sessions = sorted(set(required_tmux_sessions) - sessions)
    checks = {
        "tmux": {
            "ok": tmux_error is None and not missing_sessions,
            "sessions": sorted(sessions),
            "missing": missing_sessions,
            "error": tmux_error,
        },
        "tensorboard_http": check_url(tensorboard_url),
        "nvidia_smi": query_nvidia_smi(),
        "torch_cuda": check_torch_cuda(),
        "parallel_readiness": check_parallel_readiness(
            preset=preset,
            eval_games=eval_games,
        ),
        "resource_log": check_resource_log(
            resource_log,
            max_age_s=max_resource_age_s,
        ),
    }
    try:
        checks["tensorboard_tags"] = audit_tensorboard_logdir(tensorboard_logdir)
    except Exception as exc:
        checks["tensorboard_tags"] = {"ok": False, "error": str(exc)}
    try:
        checks["resource_tensorboard_tags"] = audit_resource_monitor_logdir(
            resource_tensorboard_logdir,
        )
    except Exception as exc:
        checks["resource_tensorboard_tags"] = {"ok": False, "error": str(exc)}

    nvidia_smi = checks["nvidia_smi"]
    nvidia_ok = "gpu_error" not in nvidia_smi
    ok = (
        checks["tmux"]["ok"]
        and checks["tensorboard_http"].get("ok", False)
        and nvidia_ok
        and checks["torch_cuda"].get("ok", False)
        and checks["parallel_readiness"].get("ok", False)
        and checks["resource_log"]["ok"]
        and checks["tensorboard_tags"].get("ok", False)
        and checks["resource_tensorboard_tags"].get("ok", False)
    )
    warnings = health_warnings(checks)
    return {
        "ok": ok,
        "warnings": warnings,
        "checks": checks,
    }


def exit_code_for_health(health, fail_on_warnings=False):
    if not health.get("ok", False):
        return 1
    if fail_on_warnings and health.get("warnings"):
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description="Check remote training monitor health.")
    parser.add_argument(
        "--tensorboard-url",
        default="http://127.0.0.1:8081",
        help="TensorBoard HTTP endpoint to check",
    )
    parser.add_argument(
        "--tensorboard-logdir",
        default="checkpoints/tensorboard/large_16x16_top_human_gpu",
        help="TensorBoard preset logdir for scalar tag audit",
    )
    parser.add_argument(
        "--resource-tensorboard-logdir",
        default="checkpoints/tensorboard/resource_monitor",
        help="TensorBoard logdir for resource_monitor.py scalar tag audit",
    )
    parser.add_argument(
        "--resource-log",
        default="resource_monitor.jsonl",
        help="resource_monitor.py JSONL output path",
    )
    parser.add_argument(
        "--preset",
        default="large_16x16_top_human_gpu",
        help="Training preset whose resolved worker counts should be checked",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=32,
        help="Evaluation game count used to cap resolved eval workers",
    )
    parser.add_argument(
        "--max-resource-age-s",
        type=float,
        default=120.0,
        help="Maximum allowed age for the latest resource sample",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Return a failing exit code when non-fatal resource warnings are present",
    )
    args = parser.parse_args()
    health = run_health_check(
        tensorboard_url=args.tensorboard_url,
        tensorboard_logdir=args.tensorboard_logdir,
        resource_tensorboard_logdir=args.resource_tensorboard_logdir,
        resource_log=args.resource_log,
        max_resource_age_s=args.max_resource_age_s,
        preset=args.preset,
        eval_games=args.eval_games,
    )
    print(json.dumps(
        health,
        indent=2,
        sort_keys=True,
    ))
    return exit_code_for_health(health, fail_on_warnings=args.fail_on_warnings)


if __name__ == "__main__":
    raise SystemExit(main())
