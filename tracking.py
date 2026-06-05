import json
import logging
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
        ):
            value = _number(event.get(key))
            if value is not None:
                scalars[tag] = value

    if kind == "self_play_game":
        for key, tag in (
            ("moves", "self_play/moves"),
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
        ):
            value = _number(event.get(key))
            if value is not None:
                scalars[tag] = value
        if "winner" in event:
            scalars["self_play/draw_rate"] = 1.0 if event.get("winner") == -1 else 0.0

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

    for key in ("buffer_size", "replay_samples"):
        value = _number(event.get(key))
        if value is not None:
            scalars["replay/samples"] = value
            break

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
    checkpoint_dir = Path(checkpoint_dir)
    log_path = checkpoint_dir / "training_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")

    scalars = tensorboard_scalars(payload)
    if not scalars:
        return

    writer = _writer(checkpoint_dir, payload.get("preset"))
    if writer is None:
        return
    step = tensorboard_step(payload)
    for tag, value in scalars.items():
        writer.add_scalar(tag, value, step)
    writer.flush()
