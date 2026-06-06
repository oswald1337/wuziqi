import argparse
import json
import re
from pathlib import Path
from statistics import mean, median

from replay_audit import audit_replay_file


def load_training_events(path, preset=None):
    events = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if preset is None or event.get("preset") == preset:
                events.append(event)
    return events


def split_runs(events):
    runs = []
    current = None
    for event in events:
        if event.get("event") == "run_start" or current is None:
            current = []
            runs.append(current)
        current.append(event)
    return runs


def select_run(runs, checkpoint_id=None):
    if not runs:
        return []
    if checkpoint_id:
        for run in reversed(runs):
            if any(event.get("checkpoint_id") == checkpoint_id for event in run):
                return run
    return runs[-1]


def _sum_number(events, key):
    return sum(float(event.get(key, 0.0) or 0.0) for event in events)


def _sum_int(events, key):
    return sum(int(event.get(key, 0) or 0) for event in events)


def _median_number(events, key):
    values = [
        float(event[key])
        for event in events
        if isinstance(event.get(key), (int, float))
    ]
    return None if not values else median(values)


def _edge_means(values, window=32):
    if not values:
        return None
    requested_window = max(1, int(window or 1))
    if len(values) >= requested_window * 2:
        effective_window = requested_window
    elif len(values) >= 2:
        effective_window = max(1, len(values) // 2)
    else:
        effective_window = 1
    first_mean = mean(values[:effective_window])
    last_mean = mean(values[-effective_window:])
    return {
        "first_mean": first_mean,
        "last_mean": last_mean,
        "delta": last_mean - first_mean,
        "window": effective_window,
        "requested_window": requested_window,
        "samples": len(values),
    }


def _loss_summary(train_events, window=32):
    summary = {}
    for key in ("loss", "policy_loss", "value_loss", "entropy"):
        values = [
            float(event[key])
            for event in train_events
            if event.get("source") is None and isinstance(event.get(key), (int, float))
        ]
        edge = _edge_means(values, window=window)
        if edge is not None:
            summary[key] = edge
    return summary


def _final_external_eval(evaluation_events):
    if not evaluation_events:
        return None
    event = evaluation_events[-1]
    opponents = event.get("evaluation", {}).get("opponents", {})
    return {
        "timestamp": event.get("timestamp"),
        "checkpoint_id": event.get("checkpoint_id"),
        "elo": event.get("elo"),
        "duration_s": event.get("duration_s"),
        "heuristic_score": opponents.get("heuristic", {}).get("score"),
        "previous_best_score": opponents.get("previous_best", {}).get("score"),
        "promoted": event.get("promotion", {}).get("promoted"),
        "failures": sum(
            int(record.get("failures", 0) or 0)
            for record in opponents.values()
            if isinstance(record, dict)
        ),
    }


def _parse_iso_timestamp(value):
    if not value:
        return None
    from datetime import datetime

    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def parse_gpu_smi_log(path, start=None, end=None):
    start_time = _parse_iso_timestamp(start)
    end_time = _parse_iso_timestamp(end)
    timestamp_pattern = re.compile(r"^===== (?P<timestamp>[^=]+) =====$")
    gpu_pattern = re.compile(
        r"\|\s*(?P<fan>\d+)%\s+"
        r"(?P<temp>\d+)C\s+"
        r"(?P<perf>P\d+)\s+.*?\|\s+"
        r"(?P<memory_used>\d+)MiB\s*/\s*"
        r"(?P<memory_total>\d+)MiB\s*\|\s+"
        r"(?P<gpu_util>\d+)%"
    )

    samples = []
    current_timestamp = None
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip()
            timestamp_match = timestamp_pattern.match(line)
            if timestamp_match:
                current_timestamp = _parse_iso_timestamp(
                    timestamp_match.group("timestamp").strip()
                )
                continue
            if current_timestamp is None:
                continue
            if start_time is not None and current_timestamp < start_time:
                continue
            if end_time is not None and current_timestamp > end_time:
                continue
            gpu_match = gpu_pattern.search(line)
            if not gpu_match:
                continue
            samples.append({
                "timestamp": current_timestamp.isoformat(),
                "gpu_util_percent": int(gpu_match.group("gpu_util")),
                "memory_used_mb": int(gpu_match.group("memory_used")),
                "memory_total_mb": int(gpu_match.group("memory_total")),
                "temperature_c": int(gpu_match.group("temp")),
                "perf_state": gpu_match.group("perf"),
            })
    return samples


def summarize_gpu_smi_samples(samples):
    if not samples:
        return {
            "samples": 0,
            "first_timestamp": None,
            "last_timestamp": None,
        }
    utils = [sample["gpu_util_percent"] for sample in samples]
    memory = [sample["memory_used_mb"] for sample in samples]
    nonzero = sum(1 for value in utils if value > 0)
    high = sum(1 for value in utils if value >= 50)
    return {
        "samples": len(samples),
        "first_timestamp": samples[0]["timestamp"],
        "last_timestamp": samples[-1]["timestamp"],
        "gpu_util_percent_mean": mean(utils),
        "gpu_util_percent_median": median(utils),
        "gpu_util_percent_max": max(utils),
        "gpu_util_nonzero_fraction": nonzero / len(samples),
        "gpu_util_high_fraction": high / len(samples),
        "gpu_util_idle_fraction": 1.0 - (nonzero / len(samples)),
        "memory_used_mb_mean": mean(memory),
        "memory_used_mb_max": max(memory),
    }


def load_resource_monitor_samples(path, start=None, end=None):
    start_time = _parse_iso_timestamp(start)
    end_time = _parse_iso_timestamp(end)
    samples = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            timestamp = _parse_iso_timestamp(sample.get("timestamp"))
            if timestamp is None:
                continue
            if start_time is not None and timestamp < start_time:
                continue
            if end_time is not None and timestamp > end_time:
                continue
            samples.append(sample)
    return samples


def _numeric_values(samples, getter):
    values = []
    for sample in samples:
        value = getter(sample)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _value_summary(values):
    if not values:
        return None
    return {
        "mean": mean(values),
        "median": median(values),
        "max": max(values),
    }


def summarize_resource_monitor_samples(samples):
    if not samples:
        return {
            "samples": 0,
            "first_timestamp": None,
            "last_timestamp": None,
        }
    cpu_values = _numeric_values(samples, lambda sample: sample.get("cpu_util_percent"))
    cpu_used_core_values = _numeric_values(samples, lambda sample: sample.get("cpu_used_cores"))
    host_cpu_values = _numeric_values(samples, lambda sample: sample.get("host_cpu_util_percent"))
    throttle_values = _numeric_values(
        samples,
        lambda sample: sample.get("cpu_throttled_period_fraction"),
    )
    load1_values = _numeric_values(samples, lambda sample: sample.get("load", {}).get("load1"))
    memory_values = _numeric_values(
        samples,
        lambda sample: sample.get("memory", {}).get("memory_used_percent"),
    )
    gpu_values = _numeric_values(
        samples,
        lambda sample: sample.get("gpu", {}).get("gpu_util_percent"),
    )
    summary = {
        "samples": len(samples),
        "first_timestamp": samples[0].get("timestamp"),
        "last_timestamp": samples[-1].get("timestamp"),
        "cpu_util_percent": _value_summary(cpu_values),
        "cpu_used_cores": _value_summary(cpu_used_core_values),
        "host_cpu_util_percent": _value_summary(host_cpu_values),
        "cpu_throttled_period_fraction": _value_summary(throttle_values),
        "load1": _value_summary(load1_values),
        "memory_used_percent": _value_summary(memory_values),
        "gpu_util_percent": _value_summary(gpu_values),
    }
    if cpu_values:
        summary["cpu_util_high_fraction"] = (
            sum(1 for value in cpu_values if value >= 80.0) / len(cpu_values)
        )
    if gpu_values:
        summary["gpu_util_idle_fraction"] = (
            sum(1 for value in gpu_values if value <= 0.0) / len(gpu_values)
        )
    return summary


def _summary_value(summary, *path):
    value = summary
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def classify_bottleneck(summary):
    evidence = []
    metrics = {}

    gpu_idle = _summary_value(summary, "gpu_smi", "gpu_util_idle_fraction")
    gpu_median = _summary_value(summary, "gpu_smi", "gpu_util_percent_median")
    gpu_mean = _summary_value(summary, "gpu_smi", "gpu_util_percent_mean")
    gpu_max = _summary_value(summary, "gpu_smi", "gpu_util_percent_max")
    if gpu_idle is None:
        gpu_idle = _summary_value(summary, "resources", "gpu_util_idle_fraction")
    if gpu_mean is None:
        gpu_mean = _summary_value(summary, "resources", "gpu_util_percent", "mean")
    if gpu_median is None:
        gpu_median = _summary_value(summary, "resources", "gpu_util_percent", "median")
    if gpu_max is None:
        gpu_max = _summary_value(summary, "resources", "gpu_util_percent", "max")

    cpu_mean = _summary_value(summary, "resources", "cpu_util_percent", "mean")
    cpu_high_fraction = _summary_value(summary, "resources", "cpu_util_high_fraction")
    cpu_used_cores_mean = _summary_value(summary, "resources", "cpu_used_cores", "mean")
    host_cpu_mean = _summary_value(summary, "resources", "host_cpu_util_percent", "mean")
    cpu_throttle_mean = _summary_value(
        summary,
        "resources",
        "cpu_throttled_period_fraction",
        "mean",
    )
    moves_per_second = summary.get("self_play_moves_per_second")
    gpu_positions_per_second = _summary_value(
        summary,
        "gpu_inference",
        "positions_per_second",
    )
    stream_elapsed = _summary_value(summary, "runtime_seconds", "stream_elapsed")
    gpu_inference_duration = _summary_value(summary, "gpu_inference", "duration_s")
    inference_active_fraction = None
    if stream_elapsed and gpu_inference_duration is not None and stream_elapsed > 0:
        inference_active_fraction = gpu_inference_duration / stream_elapsed
    parallel_wait = _summary_value(summary, "parallel_coordination", "wait_seconds")
    parallel_coalesce = _summary_value(
        summary,
        "parallel_coordination",
        "coalesce_seconds",
    )
    parallel_payload_build = _summary_value(
        summary,
        "parallel_coordination",
        "payload_build_seconds",
    )
    parallel_response_send = _summary_value(
        summary,
        "parallel_coordination",
        "response_send_seconds",
    )
    parallel_response_build = _summary_value(
        summary,
        "parallel_coordination",
        "response_build_seconds",
    )
    parallel_response_pipe_send = _summary_value(
        summary,
        "parallel_coordination",
        "response_pipe_send_seconds",
    )
    parallel_wait_fraction = None
    parallel_coalesce_fraction = None
    parallel_payload_build_fraction = None
    parallel_response_send_fraction = None
    parallel_response_build_fraction = None
    parallel_response_pipe_send_fraction = None
    if stream_elapsed and stream_elapsed > 0:
        if parallel_wait is not None:
            parallel_wait_fraction = parallel_wait / stream_elapsed
        if parallel_coalesce is not None:
            parallel_coalesce_fraction = parallel_coalesce / stream_elapsed
        if parallel_payload_build is not None:
            parallel_payload_build_fraction = parallel_payload_build / stream_elapsed
        if parallel_response_send is not None:
            parallel_response_send_fraction = parallel_response_send / stream_elapsed
        if parallel_response_build is not None:
            parallel_response_build_fraction = parallel_response_build / stream_elapsed
        if parallel_response_pipe_send is not None:
            parallel_response_pipe_send_fraction = parallel_response_pipe_send / stream_elapsed
    positions_per_request = summary.get("gpu_positions_per_request_median")
    ready_pipes_per_event = _summary_value(
        summary,
        "parallel_coordination",
        "ready_pipes_per_event_median",
    )
    coalesce_empty_wait_fraction = _summary_value(
        summary,
        "parallel_coordination",
        "coalesce_empty_wait_fraction_median",
    )
    coalesce_extra_pipes_per_call = _summary_value(
        summary,
        "parallel_coordination",
        "coalesce_extra_pipes_per_call_median",
    )

    for key, value in (
        ("gpu_idle_fraction", gpu_idle),
        ("gpu_util_percent_mean", gpu_mean),
        ("gpu_util_percent_median", gpu_median),
        ("gpu_util_percent_max", gpu_max),
        ("cpu_util_percent_mean", cpu_mean),
        ("cpu_util_high_fraction", cpu_high_fraction),
        ("cpu_used_cores_mean", cpu_used_cores_mean),
        ("host_cpu_util_percent_mean", host_cpu_mean),
        ("cpu_throttled_period_fraction_mean", cpu_throttle_mean),
        ("self_play_moves_per_second", moves_per_second),
        ("gpu_inference_positions_per_second", gpu_positions_per_second),
        ("gpu_inference_active_fraction", inference_active_fraction),
        ("gpu_inference_positions_per_request_median", positions_per_request),
        ("parallel_ready_pipes_per_event_median", ready_pipes_per_event),
        ("parallel_coalesce_empty_wait_fraction_median", coalesce_empty_wait_fraction),
        ("parallel_coalesce_extra_pipes_per_call_median", coalesce_extra_pipes_per_call),
        ("parallel_wait_fraction", parallel_wait_fraction),
        ("parallel_coalesce_fraction", parallel_coalesce_fraction),
        ("parallel_payload_build_fraction", parallel_payload_build_fraction),
        ("parallel_response_send_fraction", parallel_response_send_fraction),
        ("parallel_response_build_fraction", parallel_response_build_fraction),
        ("parallel_response_pipe_send_fraction", parallel_response_pipe_send_fraction),
    ):
        if value is not None:
            metrics[key] = value

    if gpu_idle is not None:
        evidence.append(f"GPU idle fraction {gpu_idle:.3f}")
    if gpu_mean is not None:
        evidence.append(f"GPU mean utilization {gpu_mean:.2f}%")
    if gpu_positions_per_second is not None:
        evidence.append(
            f"GPU inference throughput {gpu_positions_per_second:.1f} positions/sec"
        )
    if moves_per_second is not None:
        evidence.append(f"self-play throughput {moves_per_second:.3f} moves/sec")
    if cpu_mean is not None:
        evidence.append(f"container CPU mean utilization {cpu_mean:.2f}%")
    if cpu_used_cores_mean is not None:
        evidence.append(f"container CPU used {cpu_used_cores_mean:.2f} cores on average")
    if cpu_throttle_mean is not None and cpu_throttle_mean > 0:
        evidence.append(f"CPU throttled period fraction {cpu_throttle_mean:.3f}")
    if positions_per_request is not None:
        evidence.append(
            f"median GPU positions/request {positions_per_request:.1f}"
        )
    if ready_pipes_per_event is not None:
        evidence.append(
            f"median ready pipes/event {ready_pipes_per_event:.1f}"
        )
    if coalesce_extra_pipes_per_call is not None:
        evidence.append(
            f"coalesce extra pipes/call {coalesce_extra_pipes_per_call:.2f}"
        )
    if coalesce_empty_wait_fraction is not None:
        evidence.append(
            f"coalesce empty-wait fraction {coalesce_empty_wait_fraction:.3f}"
        )
    if parallel_wait_fraction is not None:
        evidence.append(f"parallel wait fraction {parallel_wait_fraction:.3f}")
    if parallel_coalesce_fraction is not None and parallel_coalesce_fraction > 0:
        evidence.append(f"coalesce fraction {parallel_coalesce_fraction:.3f}")

    if gpu_idle is not None and gpu_idle <= 0.20 and (gpu_median or 0.0) >= 60.0:
        label = "gpu_bound"
        confidence = "medium"
        recommendation = "Increase model batch efficiency only after confirming search is not starving the GPU."
    elif (
        cpu_mean is not None
        and cpu_high_fraction is not None
        and (cpu_mean >= 80.0 or cpu_high_fraction >= 0.5)
        and (gpu_idle is None or gpu_idle >= 0.5)
    ):
        label = "cpu_bound"
        confidence = "medium"
        recommendation = "Reduce CPU-side search overhead or rebalance worker count before scaling games."
    elif (
        gpu_idle is not None
        and gpu_idle >= 0.5
        and gpu_positions_per_second is not None
        and gpu_positions_per_second >= 100.0
        and moves_per_second is not None
        and moves_per_second < 10.0
    ):
        label = "mcts_search_coordination_bound"
        confidence = "high"
        recommendation = "Optimize MCTS/request cadence, search target quality, or batched inference scheduling before heuristic variants."
    else:
        label = "undetermined"
        confidence = "low"
        recommendation = "Collect resource_monitor.jsonl plus GPU logs during a full run before changing the recipe."

    return {
        "label": label,
        "confidence": confidence,
        "metrics": metrics,
        "evidence": evidence,
        "recommendation": recommendation,
    }


def loss_delta(summary, key):
    record = _summary_value(summary, "loss_first_last", key)
    if not isinstance(record, dict):
        return None
    first = record.get("first_mean")
    last = record.get("last_mean")
    if not isinstance(first, (int, float)) or not isinstance(last, (int, float)):
        return None
    return float(last) - float(first)


def _external_eval_improved(summary):
    eval_record = summary.get("final_external_eval") or {}
    heuristic_score = eval_record.get("heuristic_score")
    previous_best_score = eval_record.get("previous_best_score")
    if not isinstance(heuristic_score, (int, float)):
        return False
    if not isinstance(previous_best_score, (int, float)):
        return False
    return heuristic_score >= 0.5 and previous_best_score >= 0.5


def recommend_next_action(summary):
    bottleneck = summary.get("bottleneck_assessment") or {}
    replay_quality = _summary_value(
        summary,
        "replay_quality",
        "replay_quality_assessment",
    ) or {}
    promoted = bool(_summary_value(summary, "final_external_eval", "promoted"))
    loss = loss_delta(summary, "loss")
    policy_loss = loss_delta(summary, "policy_loss")
    value_loss = loss_delta(summary, "value_loss")
    entropy_delta = loss_delta(summary, "entropy")
    loss_improving = (
        loss is not None
        and policy_loss is not None
        and value_loss is not None
        and loss < 0
        and policy_loss < 0
        and value_loss < 0
    )
    evidence = []
    if bottleneck.get("label"):
        evidence.append(f"bottleneck={bottleneck['label']}")
    if replay_quality.get("label"):
        evidence.append(f"replay_quality={replay_quality['label']}")
    if loss is not None:
        evidence.append(f"loss_delta={loss:.4f}")
    if policy_loss is not None:
        evidence.append(f"policy_loss_delta={policy_loss:.4f}")
    if value_loss is not None:
        evidence.append(f"value_loss_delta={value_loss:.4f}")
    if entropy_delta is not None:
        evidence.append(f"entropy_delta={entropy_delta:.4f}")

    if promoted or _external_eval_improved(summary):
        label = "continue_same_recipe_longer"
        recommendation = "Eval scores improved; continue the same recipe with longer instrumentation."
    elif bottleneck.get("label") == "mcts_search_coordination_bound":
        if replay_quality.get("label") in {
            "diffuse_policy_targets",
            "high_entropy_policy_targets",
        }:
            label = "fix_search_target_alignment_before_scaling"
            recommendation = (
                "MCTS/search coordination is the bottleneck and replay targets are "
                "diffuse; sharpen/filter MCTS policy targets or improve search "
                "alignment before heuristic variants."
            )
        else:
            label = "optimize_mcts_or_batched_inference"
            recommendation = (
                "MCTS/search coordination is the bottleneck; optimize request cadence, "
                "batched inference, or search parallelism before scaling."
            )
    elif bottleneck.get("label") == "cpu_bound":
        label = "optimize_cpu_search_parallelism"
        recommendation = "CPU is saturated; reduce CPU-side search overhead or rebalance worker counts."
    elif bottleneck.get("label") == "gpu_bound":
        label = "optimize_gpu_batch_efficiency"
        recommendation = "GPU is saturated; improve model/batch efficiency before increasing search demand."
    elif replay_quality.get("label") == "draw_heavy_value_targets":
        label = "revise_value_targets_draw_handling"
        recommendation = "Value targets are draw-heavy; revise draw handling before longer runs."
    elif not loss_improving:
        label = "inspect_replay_or_training_signal"
        recommendation = "Losses are not consistently improving; inspect replay quality and training batches."
    else:
        label = "collect_more_instrumented_evidence"
        recommendation = "Evidence is inconclusive; run only after monitors and target-quality metrics are active."

    return {
        "label": label,
        "evidence": evidence,
        "loss_improving": loss_improving,
        "recommendation": recommendation,
    }


def summarize_run(run_events, loss_window=32):
    self_play = [event for event in run_events if event.get("event") == "self_play_game"]
    parallel = [
        event for event in run_events
        if event.get("event") == "parallel_self_play_batch"
    ]
    train = [event for event in run_events if event.get("event") == "train_step"]
    checkpoints = [
        event for event in run_events
        if event.get("event") == "checkpoint_saved"
    ]
    evaluations = [
        event for event in run_events
        if event.get("event") == "evaluation"
    ]
    logging_events = [
        event for event in run_events
        if event.get("event") == "logging_timing"
    ]

    moves = sum(int(event.get("moves", 0) or 0) for event in self_play)
    search_moves = sum(int(event.get("search_moves", 0) or 0) for event in self_play)
    stream_elapsed_s = max(
        (float(event.get("parallel_stream_elapsed_s", 0.0) or 0.0) for event in self_play),
        default=0.0,
    )
    gpu_inference_positions = _sum_int(parallel, "gpu_inference_positions")
    gpu_inference_duration_s = _sum_number(parallel, "gpu_inference_duration_s")
    final_checkpoint = checkpoints[-1] if checkpoints else {}
    parallel_wait_seconds = _sum_number(parallel, "parallel_wait_duration_s")
    parallel_coalesce_seconds = _sum_number(parallel, "parallel_coalesce_duration_s")
    parallel_payload_build_seconds = _sum_number(
        parallel,
        "parallel_payload_build_duration_s",
    )
    parallel_response_send_seconds = _sum_number(
        parallel,
        "parallel_response_send_duration_s",
    )
    parallel_response_build_seconds = _sum_number(
        parallel,
        "parallel_response_build_duration_s",
    )
    parallel_response_pipe_send_seconds = _sum_number(
        parallel,
        "parallel_response_pipe_send_duration_s",
    )

    return {
        "run_start": run_events[0].get("timestamp") if run_events else None,
        "run_end": run_events[-1].get("timestamp") if run_events else None,
        "final_checkpoint_id": final_checkpoint.get("checkpoint_id"),
        "games_trained": final_checkpoint.get("games_trained"),
        "self_play_games": len(self_play),
        "self_play_moves": moves,
        "self_play_moves_per_second": (
            None if stream_elapsed_s <= 0 else moves / stream_elapsed_s
        ),
        "draws": sum(1 for event in self_play if event.get("winner") == -1),
        "forced_tactical_moves": sum(
            int(event.get("forced_tactical_moves", 0) or 0)
            for event in self_play
        ),
        "self_play_policy_targets": {
            "samples": _sum_int(self_play, "policy_target_samples"),
            "max_prob_mean": _value_summary(_numeric_values(
                self_play,
                lambda event: event.get("policy_target_max_prob_mean"),
            )),
            "normalized_entropy_mean": _value_summary(_numeric_values(
                self_play,
                lambda event: event.get("policy_target_normalized_entropy_mean"),
            )),
            "diffuse_fraction": _value_summary(_numeric_values(
                self_play,
                lambda event: event.get("policy_target_diffuse_fraction"),
            )),
            "sharp_fraction": _value_summary(_numeric_values(
                self_play,
                lambda event: event.get("policy_target_sharp_fraction"),
            )),
            "one_hot_fraction": _value_summary(_numeric_values(
                self_play,
                lambda event: event.get("policy_target_one_hot_fraction"),
            )),
            "transform_active": _value_summary(_numeric_values(
                self_play,
                lambda event: event.get("policy_target_transform_active"),
            )),
            "transform_retained_mass": _value_summary(_numeric_values(
                self_play,
                lambda event: event.get("policy_target_transform_retained_mass_mean"),
            )),
            "transform_support_kept_fraction": _value_summary(_numeric_values(
                self_play,
                lambda event: event.get("policy_target_transform_support_kept_fraction_mean"),
            )),
            "transform_changed_top1_fraction": _value_summary(_numeric_values(
                self_play,
                lambda event: event.get("policy_target_transform_changed_top1_fraction"),
            )),
            "transform_max_prob_delta": _value_summary(_numeric_values(
                self_play,
                lambda event: event.get("policy_target_transform_max_prob_delta"),
            )),
            "transform_normalized_entropy_delta": _value_summary(_numeric_values(
                self_play,
                lambda event: event.get("policy_target_transform_normalized_entropy_delta"),
            )),
            "value_draw_fraction": _value_summary(_numeric_values(
                self_play,
                lambda event: event.get("value_target_draw_fraction"),
            )),
        },
        "search_moves": search_moves,
        "search_moves_per_second": (
            None if stream_elapsed_s <= 0 else search_moves / stream_elapsed_s
        ),
        "parallel_windows": len(parallel),
        "parallel_workers_median": _median_number(parallel, "parallel_workers"),
        "parallel_games_per_second_median": _median_number(
            parallel,
            "parallel_games_per_second",
        ),
        "parallel_coordination": {
            "wait_seconds": parallel_wait_seconds,
            "wait_calls": _sum_int(parallel, "parallel_wait_calls"),
            "wait_seconds_per_call_median": _median_number(
                parallel,
                "parallel_wait_seconds_per_call",
            ),
            "ready_events": _sum_int(parallel, "parallel_ready_events"),
            "ready_pipes": _sum_int(parallel, "parallel_ready_pipes"),
            "ready_pipes_per_event_median": _median_number(
                parallel,
                "parallel_ready_pipes_per_event",
            ),
            "messages": _sum_int(parallel, "parallel_messages"),
            "predict_messages": _sum_int(parallel, "parallel_predict_messages"),
            "game_result_messages": _sum_int(
                parallel,
                "parallel_game_result_messages",
            ),
            "coalesce_seconds": parallel_coalesce_seconds,
            "coalesce_calls": _sum_int(parallel, "parallel_coalesce_calls"),
            "coalesce_wait_calls": _sum_int(
                parallel,
                "parallel_coalesce_wait_calls",
            ),
            "coalesce_extra_pipes": _sum_int(
                parallel,
                "parallel_coalesce_extra_pipes",
            ),
            "coalesce_empty_waits": _sum_int(
                parallel,
                "parallel_coalesce_empty_waits",
            ),
            "coalesce_extra_pipes_per_call_median": _median_number(
                parallel,
                "parallel_coalesce_extra_pipes_per_call",
            ),
            "coalesce_empty_wait_fraction_median": _median_number(
                parallel,
                "parallel_coalesce_empty_wait_fraction",
            ),
            "payload_build_seconds": parallel_payload_build_seconds,
            "request_state_bytes": _sum_int(
                parallel,
                "parallel_request_state_bytes",
            ),
            "request_state_bytes_per_position_median": _median_number(
                parallel,
                "parallel_request_state_bytes_per_position",
            ),
            "request_available_values": _sum_int(
                parallel,
                "parallel_request_available_values",
            ),
            "request_available_values_per_position_median": _median_number(
                parallel,
                "parallel_request_available_values_per_position",
            ),
            "compact_requests": _sum_int(parallel, "parallel_compact_requests"),
            "compact_request_fraction_median": _median_number(
                parallel,
                "parallel_compact_request_fraction",
            ),
            "response_send_seconds": parallel_response_send_seconds,
            "response_build_seconds": parallel_response_build_seconds,
            "response_pipe_send_seconds": parallel_response_pipe_send_seconds,
            "response_probability_values": _sum_int(
                parallel,
                "parallel_response_probability_values",
            ),
            "response_probability_values_per_request_median": _median_number(
                parallel,
                "parallel_response_probability_values_per_request",
            ),
            "compact_responses": _sum_int(parallel, "parallel_compact_responses"),
            "compact_response_fraction_median": _median_number(
                parallel,
                "parallel_compact_response_fraction",
            ),
            "coalesce_fraction_median": _median_number(
                parallel,
                "parallel_coalesce_fraction",
            ),
            "payload_build_fraction_median": _median_number(
                parallel,
                "parallel_payload_build_fraction",
            ),
            "response_send_fraction_median": _median_number(
                parallel,
                "parallel_response_send_fraction",
            ),
            "response_build_fraction_median": _median_number(
                parallel,
                "parallel_response_build_fraction",
            ),
            "response_pipe_send_fraction_median": _median_number(
                parallel,
                "parallel_response_pipe_send_fraction",
            ),
        },
        "parallel_wait_seconds": parallel_wait_seconds,
        "gpu_inference": {
            "requests": _sum_int(parallel, "gpu_inference_requests"),
            "batches": _sum_int(parallel, "gpu_inference_batches"),
            "positions": gpu_inference_positions,
            "duration_s": gpu_inference_duration_s,
            "positions_per_second": (
                None
                if gpu_inference_duration_s <= 0.0
                else gpu_inference_positions / gpu_inference_duration_s
            ),
        },
        "gpu_positions_per_request_median": _median_number(
            parallel,
            "gpu_inference_positions_per_request",
        ),
        "gpu_batches_per_request_median": _median_number(
            parallel,
            "gpu_inference_batches_per_request",
        ),
        "gpu_positions_per_batch_median": _median_number(
            parallel,
            "gpu_inference_positions_per_batch",
        ),
        "gpu_positions_per_second_median": _median_number(
            parallel,
            "gpu_inference_positions_per_second",
        ),
        "runtime_seconds": {
            "stream_elapsed": stream_elapsed_s,
            "search_worker_sum": _sum_number(self_play, "search_duration_s"),
            "train": _sum_number(train, "train_duration_s"),
            "internal_eval": _sum_number(checkpoints, "eval_duration_s"),
            "external_eval": _sum_number(evaluations, "duration_s"),
            "checkpoint": _sum_number(checkpoints, "checkpoint_duration_s"),
            "replay_save": _sum_number(checkpoints, "replay_save_duration_s"),
            "parallel_wait": parallel_wait_seconds,
            "parallel_coalesce": parallel_coalesce_seconds,
            "parallel_payload_build": parallel_payload_build_seconds,
            "parallel_response_send": parallel_response_send_seconds,
            "parallel_response_build": parallel_response_build_seconds,
            "parallel_response_pipe_send": parallel_response_pipe_send_seconds,
            "logging_total": _sum_number(logging_events, "logging_duration_s"),
            "tensorboard": _sum_number(logging_events, "tensorboard_duration_s"),
            "jsonl": _sum_number(logging_events, "jsonl_write_duration_s"),
        },
        "loss_first_last": _loss_summary(train, window=loss_window),
        "final_external_eval": _final_external_eval(evaluations),
    }


def audit_training_log(path, preset=None, checkpoint_id=None, loss_window=32):
    events = load_training_events(path, preset=preset)
    run = select_run(split_runs(events), checkpoint_id=checkpoint_id)
    return summarize_run(run, loss_window=loss_window)


def attach_replay_quality(
    summary,
    replay_path=None,
    replay_max_samples=50000,
    replay_strategy="tail",
    replay_seed=0,
):
    if not replay_path:
        return summary
    summary["replay_quality"] = audit_replay_file(
        replay_path,
        max_samples=replay_max_samples,
        strategy=replay_strategy,
        seed=replay_seed,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Summarize training JSONL efficiency.")
    parser.add_argument(
        "--log",
        default="checkpoints/training_log.jsonl",
        help="Path to training_log.jsonl",
    )
    parser.add_argument("--preset", default=None, help="Preset to filter")
    parser.add_argument(
        "--checkpoint-id",
        default=None,
        help="Select the run that saved or evaluated this checkpoint",
    )
    parser.add_argument(
        "--loss-window",
        type=int,
        default=32,
        help="Number of train steps for first/last loss means",
    )
    parser.add_argument(
        "--gpu-log",
        default=None,
        help="Optional nvidia-smi monitor log to summarize over the selected run window",
    )
    parser.add_argument(
        "--resource-log",
        default=None,
        help="Optional resource_monitor.py JSONL log to summarize over the selected run window",
    )
    parser.add_argument(
        "--replay-path",
        default=None,
        help="Optional replay pickle to audit for target quality",
    )
    parser.add_argument(
        "--replay-max-samples",
        type=int,
        default=50000,
        help="Maximum replay samples to audit; 0 audits all samples",
    )
    parser.add_argument(
        "--replay-strategy",
        choices=("tail", "head", "random"),
        default="tail",
        help="Which replay slice to audit when --replay-path is set",
    )
    parser.add_argument(
        "--replay-seed",
        type=int,
        default=0,
        help="Random replay sampling seed",
    )
    args = parser.parse_args()
    summary = audit_training_log(
        args.log,
        preset=args.preset,
        checkpoint_id=args.checkpoint_id,
        loss_window=args.loss_window,
    )
    if args.gpu_log:
        summary["gpu_smi"] = summarize_gpu_smi_samples(
            parse_gpu_smi_log(
                args.gpu_log,
                start=summary.get("run_start"),
                end=summary.get("run_end"),
            )
        )
    if args.resource_log:
        summary["resources"] = summarize_resource_monitor_samples(
            load_resource_monitor_samples(
                args.resource_log,
                start=summary.get("run_start"),
                end=summary.get("run_end"),
            )
        )
    attach_replay_quality(
        summary,
        replay_path=args.replay_path,
        replay_max_samples=args.replay_max_samples,
        replay_strategy=args.replay_strategy,
        replay_seed=args.replay_seed,
    )
    summary["bottleneck_assessment"] = classify_bottleneck(summary)
    summary["decision_recommendation"] = recommend_next_action(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
