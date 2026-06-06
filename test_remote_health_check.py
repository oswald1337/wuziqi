from datetime import datetime, timezone

from remote_health_check import (
    check_resource_log,
    check_parallel_readiness,
    exit_code_for_health,
    health_warnings,
    parse_tmux_sessions,
    record_age_seconds,
    run_health_check,
)


def test_parse_tmux_sessions_extracts_session_names():
    output = (
        "gpu_monitor: 1 windows (created Fri Jun  5 18:41:39 2026)\n"
        "resource_monitor: 1 windows (created Sat Jun  6 04:09:03 2026)\n"
        "tb_wuziqi: 1 windows (created Fri Jun  5 18:37:20 2026)\n"
    )

    assert parse_tmux_sessions(output) == {
        "gpu_monitor",
        "resource_monitor",
        "tb_wuziqi",
    }


def test_record_age_seconds_uses_iso_timestamp():
    record = {"timestamp": "2026-06-06T04:00:00+00:00"}
    now = datetime(2026, 6, 6, 4, 1, 30, tzinfo=timezone.utc)

    assert record_age_seconds(record, now=now) == 90.0


def test_check_resource_log_reports_freshness(tmp_path):
    path = tmp_path / "resource_monitor.jsonl"
    path.write_text(
        '{"timestamp":"2026-06-06T04:00:00+00:00","cpu_util_percent":50.0}\n',
        encoding="utf-8",
    )

    result = check_resource_log(path, max_age_s=1.0)

    assert result["path"] == str(path)
    assert result["latest"]["cpu_util_percent"] == 50.0
    assert result["latest_timestamp"] == "2026-06-06T04:00:00+00:00"


def test_health_warnings_report_busy_cpu_baseline():
    checks = {
        "resource_log": {
            "latest": {
                "cpu_util_percent": 91.5,
                "cpu": {"capacity_cores": 12},
                "gpu": {"gpu_util_percent": 0},
            }
        },
        "nvidia_smi": {
            "gpu_util_percent": 0,
            "gpu_memory_used_mb": 0,
        },
    }

    warnings = health_warnings(checks)

    assert len(warnings) == 1
    assert "container CPU utilization is 91.5%" in warnings[0]


def test_health_warnings_report_host_cpu_without_blaming_container():
    checks = {
        "resource_log": {
            "latest": {
                "cpu_util_percent": 20.0,
                "host_cpu_util_percent": 95.0,
                "cpu": {"capacity_cores": 12},
                "gpu": {"gpu_util_percent": 0},
            }
        },
        "nvidia_smi": {
            "gpu_util_percent": 0,
            "gpu_memory_used_mb": 0,
        },
    }

    warnings = health_warnings(checks)

    assert len(warnings) == 1
    assert "host-wide CPU utilization 95.0%" in warnings[0]


def test_health_warnings_report_preexisting_gpu_activity():
    checks = {
        "resource_log": {
            "latest": {
                "cpu_util_percent": 10.0,
                "gpu": {"gpu_util_percent": 15},
            }
        },
        "nvidia_smi": {
            "gpu_util_percent": 20,
            "gpu_memory_used_mb": 512,
        },
    }

    warnings = health_warnings(checks)

    assert len(warnings) == 3
    assert any("resource monitor sees non-idle GPU" in warning for warning in warnings)
    assert any("GPU utilization 20.0%" in warning for warning in warnings)
    assert any("512MiB GPU memory" in warning for warning in warnings)


def test_health_warnings_empty_for_idle_resources():
    checks = {
        "resource_log": {
            "latest": {
                "cpu_util_percent": 20.0,
                "gpu": {"gpu_util_percent": 0},
            }
        },
        "nvidia_smi": {
            "gpu_util_percent": 0,
            "gpu_memory_used_mb": 0,
        },
    }

    assert health_warnings(checks) == []


def test_exit_code_for_health_fails_on_hard_failure():
    assert exit_code_for_health({"ok": False, "warnings": []}) == 1


def test_exit_code_for_health_allows_warnings_by_default():
    health = {"ok": True, "warnings": ["busy CPU baseline"]}

    assert exit_code_for_health(health) == 0


def test_exit_code_for_health_can_fail_on_warnings():
    health = {"ok": True, "warnings": ["busy CPU baseline"]}

    assert exit_code_for_health(health, fail_on_warnings=True) == 1


def test_check_parallel_readiness_reports_resolved_workers(monkeypatch):
    import remote_health_check

    monkeypatch.setattr(
        "config.get_training_preset",
        lambda preset: {
            "self_play_parallel_games": "auto",
            "eval_parallel_games": "auto",
            "gpu_inference_max_batch_size": 512,
            "gpu_inference_coalesce_ms": 25,
            "gpu_inference_coalesce_slice_ms": 3,
            "gpu_inference_compact_request": True,
            "gpu_inference_state_dtype": "uint8",
            "gpu_inference_compact_response": True,
            "mcts_batch_size": 128,
            "mcts_min_batches_per_search": 4,
        },
    )
    monkeypatch.setattr(
        "train._parallel_self_play_worker_count",
        lambda _config: 12,
    )
    monkeypatch.setattr(
        "train._resolve_parallel_worker_count",
        lambda _config, key, prefix, limit=None: min(limit or 12, 12),
    )

    readiness = check_parallel_readiness("large_16x16_top_human_gpu", eval_games=32)

    assert readiness["ok"] is True
    assert readiness["preset"] == "large_16x16_top_human_gpu"
    assert readiness["requested_self_play_parallel_games"] == "auto"
    assert readiness["resolved_self_play_workers"] == 12
    assert readiness["requested_eval_parallel_games"] == "auto"
    assert readiness["resolved_eval_workers"] == 12
    assert readiness["gpu_inference_coalesce_slice_ms"] == 3
    assert readiness["gpu_inference_compact_request"] is True
    assert readiness["gpu_inference_state_dtype"] == "uint8"
    assert readiness["gpu_inference_compact_response"] is True
    assert readiness["mcts_batch_size"] == 128
    assert readiness["mcts_min_batches_per_search"] == 4


def test_run_health_check_requires_resource_tensorboard_tags(monkeypatch):
    import remote_health_check

    monkeypatch.setattr(
        remote_health_check,
        "tmux_sessions",
        lambda: ({"tb_wuziqi", "gpu_monitor", "resource_monitor"}, None),
    )
    monkeypatch.setattr(
        remote_health_check,
        "check_url",
        lambda *_args, **_kwargs: {"ok": True, "status": 200},
    )
    monkeypatch.setattr(
        remote_health_check,
        "query_nvidia_smi",
        lambda: {"gpu_util_percent": 0, "gpu_memory_used_mb": 0},
    )
    monkeypatch.setattr(
        remote_health_check,
        "check_torch_cuda",
        lambda: {"ok": True, "cuda_available": True, "device_count": 1},
    )
    monkeypatch.setattr(
        remote_health_check,
        "check_parallel_readiness",
        lambda *_args, **_kwargs: {"ok": True, "resolved_self_play_workers": 12},
    )
    monkeypatch.setattr(
        remote_health_check,
        "check_resource_log",
        lambda *_args, **_kwargs: {"ok": True, "latest": {}},
    )
    monkeypatch.setattr(
        remote_health_check,
        "audit_tensorboard_logdir",
        lambda _logdir: {"ok": True},
    )
    monkeypatch.setattr(
        remote_health_check,
        "audit_resource_monitor_logdir",
        lambda _logdir: {"ok": False, "missing_required": ["resource/cpu_util_percent"]},
    )

    health = run_health_check()

    assert health["ok"] is False
    assert health["checks"]["resource_tensorboard_tags"]["missing_required"] == [
        "resource/cpu_util_percent"
    ]


def test_run_health_check_requires_torch_cuda(monkeypatch):
    import remote_health_check

    monkeypatch.setattr(
        remote_health_check,
        "tmux_sessions",
        lambda: ({"tb_wuziqi", "gpu_monitor", "resource_monitor"}, None),
    )
    monkeypatch.setattr(
        remote_health_check,
        "check_url",
        lambda *_args, **_kwargs: {"ok": True, "status": 200},
    )
    monkeypatch.setattr(
        remote_health_check,
        "query_nvidia_smi",
        lambda: {"gpu_util_percent": 0, "gpu_memory_used_mb": 0},
    )
    monkeypatch.setattr(
        remote_health_check,
        "check_torch_cuda",
        lambda: {"ok": False, "cuda_available": False, "device_count": 0},
    )
    monkeypatch.setattr(
        remote_health_check,
        "check_parallel_readiness",
        lambda *_args, **_kwargs: {"ok": True, "resolved_self_play_workers": 12},
    )
    monkeypatch.setattr(
        remote_health_check,
        "check_resource_log",
        lambda *_args, **_kwargs: {"ok": True, "latest": {}},
    )
    monkeypatch.setattr(
        remote_health_check,
        "audit_tensorboard_logdir",
        lambda _logdir: {"ok": True},
    )
    monkeypatch.setattr(
        remote_health_check,
        "audit_resource_monitor_logdir",
        lambda _logdir: {"ok": True},
    )

    health = run_health_check()

    assert health["ok"] is False
    assert health["checks"]["torch_cuda"]["cuda_available"] is False


def test_run_health_check_requires_parallel_readiness(monkeypatch):
    import remote_health_check

    monkeypatch.setattr(
        remote_health_check,
        "tmux_sessions",
        lambda: ({"tb_wuziqi", "gpu_monitor", "resource_monitor"}, None),
    )
    monkeypatch.setattr(
        remote_health_check,
        "check_url",
        lambda *_args, **_kwargs: {"ok": True, "status": 200},
    )
    monkeypatch.setattr(
        remote_health_check,
        "query_nvidia_smi",
        lambda: {"gpu_util_percent": 0, "gpu_memory_used_mb": 0},
    )
    monkeypatch.setattr(
        remote_health_check,
        "check_torch_cuda",
        lambda: {"ok": True, "cuda_available": True, "device_count": 1},
    )
    monkeypatch.setattr(
        remote_health_check,
        "check_parallel_readiness",
        lambda *_args, **_kwargs: {"ok": False, "error": "bad preset"},
    )
    monkeypatch.setattr(
        remote_health_check,
        "check_resource_log",
        lambda *_args, **_kwargs: {"ok": True, "latest": {}},
    )
    monkeypatch.setattr(
        remote_health_check,
        "audit_tensorboard_logdir",
        lambda _logdir: {"ok": True},
    )
    monkeypatch.setattr(
        remote_health_check,
        "audit_resource_monitor_logdir",
        lambda _logdir: {"ok": True},
    )

    health = run_health_check()

    assert health["ok"] is False
    assert health["checks"]["parallel_readiness"]["error"] == "bad preset"
