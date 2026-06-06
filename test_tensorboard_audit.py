from tensorboard_audit import (
    REQUIRED_TOP_HUMAN_TAGS,
    RESOURCE_MONITOR_TAGS,
    USEFUL_EFFICIENCY_TAGS,
    audit_resource_monitor_logdir,
    audit_tensorboard_tags,
)


def test_tensorboard_audit_passes_when_required_tags_present():
    tags = REQUIRED_TOP_HUMAN_TAGS | {"extra/tag"}

    result = audit_tensorboard_tags(tags, useful=set())

    assert result["ok"] is True
    assert result["missing_required"] == []
    assert set(result["present_required"]) == REQUIRED_TOP_HUMAN_TAGS


def test_tensorboard_audit_reports_missing_required_and_useful_tags():
    tags = {"train/loss", "runtime/gpu_inference_batches"}

    result = audit_tensorboard_tags(tags)

    assert result["ok"] is False
    assert "train/policy_loss" in result["missing_required"]
    assert "runtime/gpu_inference_positions" in result["missing_useful"]
    assert "runtime/gpu_inference_batches" in result["present_useful"]


def test_tensorboard_required_tags_match_top_human_tracking_contract():
    expected = {
        "train/loss",
        "train/policy_loss",
        "train/value_loss",
        "train/entropy",
        "self_play/moves",
        "self_play/draw_rate",
        "self_play/forced_tactical_moves",
        "self_play/search_moves",
        "eval/elo",
        "eval/heuristic_score",
        "eval/previous_best_score",
        "replay/samples",
        "runtime/search_seconds",
        "runtime/train_seconds",
        "runtime/eval_seconds",
        "runtime/logging_seconds",
        "runtime/checkpoint_seconds",
    }

    assert REQUIRED_TOP_HUMAN_TAGS == expected
    assert "runtime/gpu_inference_positions_per_second" in USEFUL_EFFICIENCY_TAGS
    assert "runtime/gpu_inference_positions_per_request" in USEFUL_EFFICIENCY_TAGS
    assert "runtime/parallel_coalesce_seconds" in USEFUL_EFFICIENCY_TAGS
    assert "runtime/parallel_coalesce_extra_pipes_per_call" in USEFUL_EFFICIENCY_TAGS
    assert "runtime/parallel_coalesce_empty_wait_fraction" in USEFUL_EFFICIENCY_TAGS
    assert "runtime/parallel_payload_build_seconds" in USEFUL_EFFICIENCY_TAGS
    assert "runtime/parallel_response_send_seconds" in USEFUL_EFFICIENCY_TAGS
    assert "runtime/parallel_response_build_seconds" in USEFUL_EFFICIENCY_TAGS
    assert "runtime/parallel_response_pipe_send_seconds" in USEFUL_EFFICIENCY_TAGS
    assert "runtime/parallel_compact_response_fraction" in USEFUL_EFFICIENCY_TAGS
    assert "runtime/parallel_request_state_bytes_per_position" in USEFUL_EFFICIENCY_TAGS
    assert "runtime/parallel_compact_request_fraction" in USEFUL_EFFICIENCY_TAGS
    assert "runtime/parallel_ready_pipes_per_event" in USEFUL_EFFICIENCY_TAGS
    assert "self_play/effective_mcts_batch_size" in USEFUL_EFFICIENCY_TAGS
    assert "self_play/policy_target_diffuse_fraction" in USEFUL_EFFICIENCY_TAGS
    assert "self_play/policy_target_transform_retained_mass_mean" in USEFUL_EFFICIENCY_TAGS
    assert "self_play/policy_target_transform_normalized_entropy_delta" in USEFUL_EFFICIENCY_TAGS
    assert "self_play/value_target_draw_fraction" in USEFUL_EFFICIENCY_TAGS


def test_resource_monitor_tags_match_cpu_gpu_tracking_contract():
    expected = {
        "resource/cpu_util_percent",
        "resource/cpu_used_cores",
        "resource/cpu_capacity_cores",
        "resource/cpu_quota_cores",
        "resource/cpu_usable_workers",
        "resource/host_cpu_util_percent",
        "resource/cpu_throttled_period_fraction",
        "resource/memory_used_percent",
        "resource/memory_used_mb",
        "resource/gpu_util_percent",
        "resource/gpu_memory_used_mb",
        "resource/gpu_temperature_c",
    }

    assert RESOURCE_MONITOR_TAGS == expected


def test_resource_monitor_logdir_audit_uses_resource_tags(monkeypatch):
    import tensorboard_audit

    monkeypatch.setattr(
        tensorboard_audit,
        "scalar_tags",
        lambda _logdir: RESOURCE_MONITOR_TAGS,
    )

    result = audit_resource_monitor_logdir("checkpoints/tensorboard/resource_monitor")

    assert result["ok"] is True
    assert result["missing_required"] == []
    assert set(result["present_required"]) == RESOURCE_MONITOR_TAGS
