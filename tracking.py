import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path


_SUMMARY_WRITERS = {}
_SUMMARY_WRITER_CLASS = None
_SUMMARY_WRITER_IMPORT_ATTEMPTED = False


def _safe_name(value):
    text = str(value or "default")
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in text)


def _summary_writer_class():
    global _SUMMARY_WRITER_CLASS, _SUMMARY_WRITER_IMPORT_ATTEMPTED
    if _SUMMARY_WRITER_IMPORT_ATTEMPTED:
        return _SUMMARY_WRITER_CLASS
    _SUMMARY_WRITER_IMPORT_ATTEMPTED = True
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        logging.getLogger("tracking").warning(
            "TensorBoard writer unavailable; install project dependencies to enable charts: %s",
            exc,
        )
        _SUMMARY_WRITER_CLASS = None
    else:
        _SUMMARY_WRITER_CLASS = SummaryWriter
    return _SUMMARY_WRITER_CLASS


def _writer(checkpoint_dir, preset):
    writer_class = _summary_writer_class()
    if writer_class is None:
        return None
    log_dir = Path(checkpoint_dir) / "tensorboard" / _safe_name(preset)
    key = log_dir.as_posix()
    if key not in _SUMMARY_WRITERS:
        log_dir.mkdir(parents=True, exist_ok=True)
        _SUMMARY_WRITERS[key] = writer_class(log_dir=str(log_dir))
    return _SUMMARY_WRITERS[key]


def _number(value):
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _score_from_eval_record(record):
    if not isinstance(record, dict):
        return None
    return _number(record.get("score"))


def _opponents_from_event(event):
    evaluation = event.get("evaluation")
    if isinstance(evaluation, dict):
        opponents = evaluation.get("opponents")
        if isinstance(opponents, dict):
            return opponents
    eval_record = event.get("eval")
    if isinstance(eval_record, dict):
        return eval_record
    return {}


def tensorboard_scalars(event):
    scalars = {}
    kind = event.get("event")

    if kind == "train_step":
        prefix_by_source = {
            "conversion_replay": "conversion",
            "conversion_teacher": "conversion_teacher",
            "mcts_distill": "mcts_distill",
            "threat_space_puzzle": "threat_space",
            "threat_space_proof": "proof_value",
        }
        prefix = prefix_by_source.get(event.get("source"), "train")
        for key, tag in (
            ("loss", f"{prefix}/loss"),
            ("policy_loss", f"{prefix}/policy_loss"),
            ("value_loss", f"{prefix}/value_loss"),
            ("entropy", f"{prefix}/entropy"),
            ("priority_samples", f"{prefix}/priority_samples"),
            ("train_duration_s", "runtime/train_seconds"),
            ("gpu_memory_allocated_mb", "runtime/gpu_memory_allocated_mb"),
            ("gpu_memory_reserved_mb", "runtime/gpu_memory_reserved_mb"),
            ("gpu_max_memory_allocated_mb", "runtime/gpu_max_memory_allocated_mb"),
        ):
            value = _number(event.get(key))
            if value is not None:
                scalars[tag] = value

    if kind == "self_play_game":
        for key, tag in (
            ("moves", "self_play/moves"),
            ("parallel_workers", "self_play/parallel_workers"),
            ("forced_tactical_moves", "self_play/forced_tactical_moves"),
            ("threat_solver_moves", "self_play/threat_solver_moves"),
            ("two_ply_threat_moves", "self_play/two_ply_threat_moves"),
            ("search_moves", "self_play/search_moves"),
            ("tactical_prior_searches", "self_play/tactical_prior_searches"),
            ("tactical_prior_two_ply_hits", "self_play/tactical_prior_two_ply_hits"),
            ("tactical_leaf_evaluations", "self_play/tactical_leaf_evaluations"),
            ("tactical_leaf_positive", "self_play/tactical_leaf_positive"),
            ("tactical_leaf_negative", "self_play/tactical_leaf_negative"),
            ("tactical_leaf_win", "self_play/tactical_leaf_win"),
            ("tactical_leaf_forcing_win", "self_play/tactical_leaf_forcing_win"),
            ("tactical_leaf_two_ply_threat", "self_play/tactical_leaf_two_ply_threat"),
            ("tactical_leaf_multiple_immediate_losses", "self_play/tactical_leaf_multiple_immediate_losses"),
            ("dirichlet_noise_moves", "self_play/dirichlet_noise_moves"),
            ("no_noise_moves", "self_play/no_noise_moves"),
            ("batched_policy_batches", "self_play/batched_policy_batches"),
            ("batched_policy_positions", "self_play/batched_policy_positions"),
            ("effective_mcts_batch_size", "self_play/effective_mcts_batch_size"),
            ("moves_per_second", "self_play/moves_per_second"),
            ("policy_target_samples", "self_play/policy_target_samples"),
            ("policy_target_invalid_samples", "self_play/policy_target_invalid_samples"),
            ("policy_target_max_prob_mean", "self_play/policy_target_max_prob_mean"),
            ("policy_target_max_prob_median", "self_play/policy_target_max_prob_median"),
            ("policy_target_entropy_mean", "self_play/policy_target_entropy_mean"),
            ("policy_target_entropy_median", "self_play/policy_target_entropy_median"),
            ("policy_target_normalized_entropy_mean", "self_play/policy_target_normalized_entropy_mean"),
            ("policy_target_normalized_entropy_median", "self_play/policy_target_normalized_entropy_median"),
            ("policy_target_nonzero_actions_mean", "self_play/policy_target_nonzero_actions_mean"),
            ("policy_target_diffuse_fraction", "self_play/policy_target_diffuse_fraction"),
            ("policy_target_sharp_fraction", "self_play/policy_target_sharp_fraction"),
            ("policy_target_one_hot_fraction", "self_play/policy_target_one_hot_fraction"),
            ("policy_target_transform_active", "self_play/policy_target_transform_active"),
            ("policy_target_transform_invalid_policy_samples", "self_play/policy_target_transform_invalid_policy_samples"),
            ("policy_target_transform_retained_mass_mean", "self_play/policy_target_transform_retained_mass_mean"),
            ("policy_target_transform_support_kept_fraction_mean", "self_play/policy_target_transform_support_kept_fraction_mean"),
            ("policy_target_transform_changed_top1_fraction", "self_play/policy_target_transform_changed_top1_fraction"),
            ("policy_target_transform_fallback_top1_fraction", "self_play/policy_target_transform_fallback_top1_fraction"),
            ("policy_target_transform_max_prob_delta", "self_play/policy_target_transform_max_prob_delta"),
            ("policy_target_transform_normalized_entropy_delta", "self_play/policy_target_transform_normalized_entropy_delta"),
            ("value_target_positive_fraction", "self_play/value_target_positive_fraction"),
            ("value_target_negative_fraction", "self_play/value_target_negative_fraction"),
            ("value_target_draw_fraction", "self_play/value_target_draw_fraction"),
            ("duration_s", "runtime/self_play_seconds"),
            ("search_duration_s", "runtime/search_seconds"),
            ("train_duration_s", "runtime/train_seconds"),
        ):
            value = _number(event.get(key))
            if value is not None:
                scalars[tag] = value
        search_duration = _number(event.get("search_duration_s"))
        search_moves = _number(event.get("search_moves"))
        if search_duration is not None and search_moves and search_moves > 0:
            scalars["runtime/search_seconds_per_move"] = search_duration / search_moves
        if "winner" in event:
            scalars["self_play/draw_rate"] = 1.0 if event.get("winner") == -1 else 0.0

    if kind == "parallel_self_play_batch":
        for key, tag in (
            ("parallel_workers", "self_play/parallel_workers"),
            ("parallel_batch_games", "self_play/parallel_batch_games"),
            ("parallel_games_per_second", "self_play/parallel_games_per_second"),
            ("parallel_batch_duration_s", "runtime/parallel_self_play_batch_seconds"),
            ("parallel_wait_duration_s", "runtime/parallel_wait_seconds"),
            ("parallel_wait_calls", "runtime/parallel_wait_calls"),
            ("parallel_wait_seconds_per_call", "runtime/parallel_wait_seconds_per_call"),
            ("parallel_ready_events", "runtime/parallel_ready_events"),
            ("parallel_ready_pipes", "runtime/parallel_ready_pipes"),
            ("parallel_ready_pipes_per_event", "runtime/parallel_ready_pipes_per_event"),
            ("parallel_messages", "runtime/parallel_messages"),
            ("parallel_predict_messages", "runtime/parallel_predict_messages"),
            ("parallel_game_result_messages", "runtime/parallel_game_result_messages"),
            ("parallel_coalesce_duration_s", "runtime/parallel_coalesce_seconds"),
            ("parallel_coalesce_calls", "runtime/parallel_coalesce_calls"),
            ("parallel_coalesce_wait_calls", "runtime/parallel_coalesce_wait_calls"),
            ("parallel_coalesce_extra_pipes", "runtime/parallel_coalesce_extra_pipes"),
            ("parallel_coalesce_extra_pipes_per_call", "runtime/parallel_coalesce_extra_pipes_per_call"),
            ("parallel_coalesce_empty_waits", "runtime/parallel_coalesce_empty_waits"),
            ("parallel_coalesce_empty_wait_fraction", "runtime/parallel_coalesce_empty_wait_fraction"),
            ("parallel_coalesce_fraction", "runtime/parallel_coalesce_fraction"),
            ("parallel_payload_build_duration_s", "runtime/parallel_payload_build_seconds"),
            ("parallel_payload_build_fraction", "runtime/parallel_payload_build_fraction"),
            ("parallel_request_state_bytes", "runtime/parallel_request_state_bytes"),
            ("parallel_request_state_bytes_per_position", "runtime/parallel_request_state_bytes_per_position"),
            ("parallel_request_available_values", "runtime/parallel_request_available_values"),
            ("parallel_request_available_values_per_position", "runtime/parallel_request_available_values_per_position"),
            ("parallel_compact_requests", "runtime/parallel_compact_requests"),
            ("parallel_compact_request_fraction", "runtime/parallel_compact_request_fraction"),
            ("parallel_response_send_duration_s", "runtime/parallel_response_send_seconds"),
            ("parallel_response_send_fraction", "runtime/parallel_response_send_fraction"),
            ("parallel_response_build_duration_s", "runtime/parallel_response_build_seconds"),
            ("parallel_response_build_fraction", "runtime/parallel_response_build_fraction"),
            ("parallel_response_pipe_send_duration_s", "runtime/parallel_response_pipe_send_seconds"),
            ("parallel_response_pipe_send_fraction", "runtime/parallel_response_pipe_send_fraction"),
            ("parallel_response_probability_values", "runtime/parallel_response_probability_values"),
            ("parallel_response_probability_values_per_request", "runtime/parallel_response_probability_values_per_request"),
            ("parallel_compact_responses", "runtime/parallel_compact_responses"),
            ("parallel_compact_response_fraction", "runtime/parallel_compact_response_fraction"),
            ("gpu_inference_requests", "runtime/gpu_inference_requests"),
            ("gpu_inference_batches", "runtime/gpu_inference_batches"),
            ("gpu_inference_positions", "runtime/gpu_inference_positions"),
            ("gpu_inference_duration_s", "runtime/gpu_inference_seconds"),
            ("gpu_inference_positions_per_batch", "runtime/gpu_inference_positions_per_batch"),
            ("gpu_inference_positions_per_request", "runtime/gpu_inference_positions_per_request"),
            ("gpu_inference_batches_per_request", "runtime/gpu_inference_batches_per_request"),
            ("gpu_inference_positions_per_second", "runtime/gpu_inference_positions_per_second"),
        ):
            value = _number(event.get(key))
            if value is not None:
                scalars[tag] = value

    if kind == "runtime_budget":
        for key, tag in (
            ("max_runtime_s", "runtime/budget_max_seconds"),
            ("runtime_dispatch_margin_s", "runtime/budget_dispatch_margin_seconds"),
            ("runtime_elapsed_s", "runtime/budget_elapsed_seconds"),
            ("runtime_remaining_s", "runtime/budget_remaining_seconds"),
            ("self_play_requested_games", "self_play/requested_games"),
            ("self_play_dispatched_games", "self_play/dispatched_games"),
            ("self_play_completed_games", "self_play/completed_games"),
            ("self_play_remaining_games", "self_play/remaining_games"),
            ("self_play_stream_elapsed_s", "runtime/self_play_stream_seconds"),
            ("self_play_dispatch_stop_estimated_game_s", "runtime/dispatch_stop_estimated_game_seconds"),
        ):
            value = _number(event.get(key))
            if value is not None:
                scalars[tag] = value
        if "runtime_budget_exceeded" in event:
            scalars["runtime/budget_exceeded"] = (
                1.0 if event.get("runtime_budget_exceeded") else 0.0
            )
        if "self_play_stopped_early" in event:
            scalars["self_play/stopped_early"] = (
                1.0 if event.get("self_play_stopped_early") else 0.0
            )
        if "runtime_dispatch_stopped" in event:
            scalars["runtime/dispatch_stopped"] = (
                1.0 if event.get("runtime_dispatch_stopped") else 0.0
            )

    if kind in {
        "bootstrap_game",
        "tactical_puzzles",
        "hard_position_puzzles",
        "threat_space_puzzles",
        "threat_space_proof_values",
        "mcts_distill_positions",
    }:
        duration = _number(event.get("duration_s"))
        if duration is not None:
            source = _safe_name(event.get("source") or kind)
            scalars[f"runtime/{source}_seconds"] = duration

    elo = _number(event.get("elo"))
    if elo is not None and kind in {"checkpoint_saved", "evaluation"}:
        scalars["eval/elo"] = elo

    opponents = _opponents_from_event(event)
    heuristic_score = _score_from_eval_record(opponents.get("heuristic"))
    if heuristic_score is not None:
        scalars["eval/heuristic_score"] = heuristic_score
    previous_best_score = _score_from_eval_record(opponents.get("previous_best"))
    if previous_best_score is not None:
        scalars["eval/previous_best_score"] = previous_best_score

    if kind in {"evaluation_opponent_start", "evaluation_opponent"}:
        opponent_key = _safe_name(event.get("opponent_key") or "opponent")
        for key, tag in (
            ("games", f"eval/{opponent_key}_games"),
            ("parallel_workers", f"eval/{opponent_key}_parallel_workers"),
            ("n_playout", f"eval/{opponent_key}_candidate_playouts"),
            ("opponent_n_playout", f"eval/{opponent_key}_opponent_playouts"),
            ("evaluation_elapsed_s", "runtime/eval_elapsed_seconds"),
        ):
            value = _number(event.get(key))
            if value is not None:
                scalars[tag] = value
        if kind == "evaluation_opponent_start":
            scalars[f"eval/{opponent_key}_started"] = 1.0
        else:
            for key, tag in (
                ("score", f"eval/{opponent_key}_score"),
                ("wins", f"eval/{opponent_key}_wins"),
                ("draws", f"eval/{opponent_key}_draws"),
                ("losses", f"eval/{opponent_key}_losses"),
                ("failures", f"eval/{opponent_key}_failures"),
                ("duration_s", f"runtime/eval_{opponent_key}_seconds"),
                ("avg_moves", f"eval/{opponent_key}_avg_moves"),
                ("win_avg_moves", f"eval/{opponent_key}_win_avg_moves"),
                ("draw_avg_moves", f"eval/{opponent_key}_draw_avg_moves"),
                ("loss_avg_moves", f"eval/{opponent_key}_loss_avg_moves"),
                ("opponents_completed", "eval/opponents_completed"),
            ):
                value = _number(event.get(key))
                if value is not None:
                    scalars[tag] = value
            score = _number(event.get("score"))
            if score is not None:
                if opponent_key == "heuristic":
                    scalars["eval/heuristic_score"] = score
                elif opponent_key == "previous_best":
                    scalars["eval/previous_best_score"] = score

    for key in ("buffer_size", "replay_samples"):
        value = _number(event.get(key))
        if value is not None:
            scalars["replay/samples"] = value
            break

    for key, tag in (
        ("checkpoint_duration_s", "runtime/checkpoint_seconds"),
        ("replay_save_duration_s", "runtime/replay_save_seconds"),
        ("eval_duration_s", "runtime/eval_seconds"),
    ):
        value = _number(event.get(key))
        if value is not None:
            scalars[tag] = value
    if kind == "evaluation":
        duration = _number(event.get("duration_s"))
        if duration is not None:
            scalars["runtime/eval_seconds"] = duration
    if kind == "logging_timing":
        for key, tag in (
            ("logging_duration_s", "runtime/logging_seconds"),
            ("jsonl_write_duration_s", "runtime/jsonl_write_seconds"),
            ("tensorboard_duration_s", "runtime/tensorboard_seconds"),
        ):
            value = _number(event.get(key))
            if value is not None:
                scalars[tag] = value

    conversion_samples = _number(event.get("conversion_replay_samples"))
    if conversion_samples is not None:
        scalars["replay/conversion_samples"] = conversion_samples
    teacher_samples = _number(event.get("conversion_teacher_samples"))
    if teacher_samples is not None:
        scalars["replay/conversion_teacher_samples"] = teacher_samples
    distill_samples = _number(event.get("mcts_distill_samples"))
    if distill_samples is not None:
        scalars["replay/mcts_distill_samples"] = distill_samples
    proof_samples = _number(event.get("threat_space_proof_samples"))
    if proof_samples is not None:
        scalars["proof_value/samples"] = proof_samples
    for key, tag in (
        ("threat_space_proof_roots", "proof_value/roots"),
        ("threat_space_proof_defender_states", "proof_value/defender_states"),
        ("threat_space_proof_replies", "proof_value/replies"),
        ("threat_space_proof_followups", "proof_value/followups"),
        ("threat_space_proof_skipped", "proof_value/skipped"),
    ):
        value = _number(event.get(key))
        if value is not None:
            scalars[tag] = value

    for key, tag in (
        ("mcts_distill_attempts", "mcts_distill/attempts"),
        ("mcts_distill_skipped", "mcts_distill/skipped"),
        ("mcts_distill_accept_rate", "mcts_distill/accept_rate"),
        ("mcts_distill_target_mass", "mcts_distill/target_mass"),
        ("mcts_distill_target_top_rate", "mcts_distill/target_top_rate"),
        ("mcts_distill_search_target_mass", "mcts_distill/search_target_mass"),
        ("mcts_distill_search_top_rate", "mcts_distill/search_top_rate"),
        ("mcts_distill_entropy", "mcts_distill/target_entropy"),
        ("mcts_distill_leaf_evaluations", "mcts_distill/leaf_evaluations"),
        ("mcts_distill_leaf_positive", "mcts_distill/leaf_positive"),
        ("mcts_distill_leaf_negative", "mcts_distill/leaf_negative"),
    ):
        value = _number(event.get(key))
        if value is not None:
            scalars[tag] = value

    source_stats = event.get("mcts_distill_source_stats")
    if isinstance(source_stats, dict):
        for source, stats in source_stats.items():
            if not isinstance(stats, dict):
                continue
            source_slug = str(source).replace("/", "_")
            for key, suffix in (
                ("samples", "samples"),
                ("attempts", "attempts"),
                ("skipped", "skipped"),
                ("accept_rate", "accept_rate"),
                ("target_mass", "target_mass"),
                ("target_top_rate", "target_top_rate"),
                ("search_target_mass", "search_target_mass"),
                ("search_top_rate", "search_top_rate"),
                ("target_entropy", "target_entropy"),
            ):
                value = _number(stats.get(key))
                if value is not None:
                    scalars[f"mcts_distill/{source_slug}/{suffix}"] = value

    return scalars


def tensorboard_step(event):
    for key in ("total_games", "games_trained", "game", "games"):
        value = event.get(key)
        if isinstance(value, int):
            return value
    return 0


def append_training_event(event, checkpoint_dir="checkpoints"):
    logging_start = time.perf_counter()
    checkpoint_dir = Path(checkpoint_dir)
    log_path = checkpoint_dir / "training_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    write_start = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
    jsonl_write_duration_s = time.perf_counter() - write_start

    scalars = tensorboard_scalars(payload)
    tensorboard_duration_s = 0.0
    writer = _writer(checkpoint_dir, payload.get("preset"))
    step = tensorboard_step(payload)
    if scalars and writer is not None:
        tensorboard_start = time.perf_counter()
        for tag, value in scalars.items():
            writer.add_scalar(tag, value, step)
        writer.flush()
        tensorboard_duration_s = time.perf_counter() - tensorboard_start

    if payload.get("event") == "logging_timing":
        return

    timing_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "logging_timing",
        "preset": payload.get("preset"),
        "source_event": payload.get("event"),
        "logging_duration_s": round(time.perf_counter() - logging_start, 6),
        "jsonl_write_duration_s": round(jsonl_write_duration_s, 6),
        "tensorboard_duration_s": round(tensorboard_duration_s, 6),
    }
    for key in ("total_games", "games_trained", "game", "games"):
        if isinstance(payload.get(key), int):
            timing_payload[key] = payload[key]
    with log_path.open("a", encoding="utf-8") as handle:
        json.dump(timing_payload, handle, sort_keys=True)
        handle.write("\n")
    timing_scalars = tensorboard_scalars(timing_payload)
    if timing_scalars and writer is not None:
        for tag, value in timing_scalars.items():
            writer.add_scalar(tag, value, step)
        writer.flush()
