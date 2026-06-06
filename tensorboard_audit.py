import argparse
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


REQUIRED_TOP_HUMAN_TAGS = {
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


USEFUL_EFFICIENCY_TAGS = {
    "self_play/moves_per_second",
    "self_play/parallel_workers",
    "self_play/parallel_games_per_second",
    "self_play/effective_mcts_batch_size",
    "runtime/gpu_inference_requests",
    "runtime/gpu_inference_batches",
    "runtime/gpu_inference_positions",
    "runtime/gpu_inference_seconds",
    "runtime/gpu_inference_positions_per_batch",
    "runtime/gpu_inference_positions_per_request",
    "runtime/gpu_inference_positions_per_second",
    "runtime/jsonl_write_seconds",
    "runtime/tensorboard_seconds",
    "runtime/replay_save_seconds",
    "runtime/parallel_wait_seconds",
    "runtime/parallel_wait_calls",
    "runtime/parallel_wait_seconds_per_call",
    "runtime/parallel_ready_pipes_per_event",
    "runtime/parallel_coalesce_seconds",
    "runtime/parallel_coalesce_calls",
    "runtime/parallel_coalesce_extra_pipes_per_call",
    "runtime/parallel_coalesce_empty_wait_fraction",
    "runtime/parallel_payload_build_seconds",
    "runtime/parallel_request_state_bytes",
    "runtime/parallel_request_state_bytes_per_position",
    "runtime/parallel_request_available_values",
    "runtime/parallel_request_available_values_per_position",
    "runtime/parallel_compact_request_fraction",
    "runtime/parallel_response_send_seconds",
    "runtime/parallel_response_build_seconds",
    "runtime/parallel_response_pipe_send_seconds",
    "runtime/parallel_compact_response_fraction",
    "runtime/parallel_self_play_batch_seconds",
    "runtime/eval_random_seconds",
    "runtime/eval_heuristic_seconds",
    "runtime/eval_previous_best_seconds",
    "runtime/budget_dispatch_margin_seconds",
    "runtime/dispatch_stop_estimated_game_seconds",
    "runtime/dispatch_stopped",
    "eval/random_score",
    "eval/opponents_completed",
    "self_play/policy_target_samples",
    "self_play/policy_target_max_prob_mean",
    "self_play/policy_target_entropy_mean",
    "self_play/policy_target_normalized_entropy_mean",
    "self_play/policy_target_diffuse_fraction",
    "self_play/policy_target_sharp_fraction",
    "self_play/policy_target_one_hot_fraction",
    "self_play/policy_target_transform_active",
    "self_play/policy_target_transform_retained_mass_mean",
    "self_play/policy_target_transform_support_kept_fraction_mean",
    "self_play/policy_target_transform_changed_top1_fraction",
    "self_play/policy_target_transform_max_prob_delta",
    "self_play/policy_target_transform_normalized_entropy_delta",
    "self_play/value_target_draw_fraction",
}


RESOURCE_MONITOR_TAGS = {
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


def scalar_tags(logdir):
    accumulator = EventAccumulator(str(logdir))
    accumulator.Reload()
    return set(accumulator.Tags().get("scalars", []))


def audit_tensorboard_tags(tags, required=None, useful=None):
    tags = set(tags)
    required = set(REQUIRED_TOP_HUMAN_TAGS if required is None else required)
    useful = set(USEFUL_EFFICIENCY_TAGS if useful is None else useful)
    missing_required = sorted(required - tags)
    missing_useful = sorted(useful - tags)
    return {
        "ok": not missing_required,
        "scalar_tag_count": len(tags),
        "missing_required": missing_required,
        "missing_useful": missing_useful,
        "present_required": sorted(required & tags),
        "present_useful": sorted(useful & tags),
    }


def audit_tensorboard_logdir(logdir):
    tags = scalar_tags(logdir)
    result = audit_tensorboard_tags(tags)
    result["logdir"] = str(logdir)
    return result


def audit_resource_monitor_logdir(logdir):
    tags = scalar_tags(logdir)
    result = audit_tensorboard_tags(
        tags,
        required=RESOURCE_MONITOR_TAGS,
        useful=set(),
    )
    result["logdir"] = str(logdir)
    return result


def main():
    parser = argparse.ArgumentParser(description="Audit TensorBoard scalar tags.")
    parser.add_argument(
        "--logdir",
        default="checkpoints/tensorboard",
        help="TensorBoard log directory or preset subdirectory",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="Preset subdirectory under --logdir",
    )
    args = parser.parse_args()
    logdir = Path(args.logdir)
    if args.preset:
        logdir = logdir / args.preset
    print(json.dumps(audit_tensorboard_logdir(logdir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
