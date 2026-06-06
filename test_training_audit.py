import pickle

import numpy as np

from training_audit import (
    attach_replay_quality,
    audit_training_log,
    classify_bottleneck,
    loss_delta,
    load_resource_monitor_samples,
    parse_gpu_smi_log,
    recommend_next_action,
    split_runs,
    summarize_gpu_smi_samples,
    summarize_resource_monitor_samples,
    summarize_run,
)


def test_split_runs_starts_new_group_on_run_start():
    events = [
        {"event": "run_start", "timestamp": "a", "preset": "p"},
        {"event": "self_play_game", "preset": "p"},
        {"event": "run_start", "timestamp": "b", "preset": "p"},
        {"event": "checkpoint_saved", "preset": "p"},
    ]

    runs = split_runs(events)

    assert len(runs) == 2
    assert runs[0][0]["timestamp"] == "a"
    assert runs[1][0]["timestamp"] == "b"


def test_summarize_run_reports_efficiency_and_eval_metrics():
    run = [
        {"event": "run_start", "timestamp": "start", "preset": "p"},
        {
            "event": "self_play_game",
            "preset": "p",
            "winner": 1,
            "moves": 10,
            "search_moves": 8,
            "forced_tactical_moves": 2,
            "parallel_stream_elapsed_s": 5.0,
            "search_duration_s": 7.5,
            "policy_target_samples": 10,
            "policy_target_max_prob_mean": 0.12,
            "policy_target_normalized_entropy_mean": 0.8,
            "policy_target_diffuse_fraction": 0.6,
            "policy_target_sharp_fraction": 0.2,
            "policy_target_one_hot_fraction": 0.1,
            "policy_target_transform_active": 1.0,
            "policy_target_transform_retained_mass_mean": 0.25,
            "policy_target_transform_support_kept_fraction_mean": 0.15,
            "policy_target_transform_changed_top1_fraction": 0.0,
            "policy_target_transform_max_prob_delta": 0.06,
            "policy_target_transform_normalized_entropy_delta": -0.36,
            "value_target_draw_fraction": 0.0,
        },
        {
            "event": "parallel_self_play_batch",
            "preset": "p",
            "parallel_workers": 2,
            "parallel_games_per_second": 0.2,
            "parallel_wait_duration_s": 3.0,
            "parallel_wait_calls": 4,
            "parallel_wait_seconds_per_call": 0.75,
            "parallel_ready_events": 2,
            "parallel_ready_pipes": 5,
            "parallel_ready_pipes_per_event": 2.5,
            "parallel_messages": 6,
            "parallel_predict_messages": 4,
            "parallel_game_result_messages": 2,
            "parallel_coalesce_duration_s": 0.25,
            "parallel_coalesce_calls": 2,
            "parallel_coalesce_wait_calls": 3,
            "parallel_coalesce_extra_pipes": 3,
            "parallel_coalesce_extra_pipes_per_call": 1.5,
            "parallel_coalesce_empty_waits": 1,
            "parallel_coalesce_empty_wait_fraction": 1 / 3,
            "parallel_coalesce_fraction": 0.05,
            "parallel_payload_build_duration_s": 0.1,
            "parallel_payload_build_fraction": 0.02,
            "parallel_request_state_bytes": 24 * 4 * 16 * 16,
            "parallel_request_state_bytes_per_position": 4 * 16 * 16,
            "parallel_request_available_values": 512,
            "parallel_request_available_values_per_position": 512 / 24,
            "parallel_compact_requests": 4,
            "parallel_compact_request_fraction": 1.0,
            "parallel_response_send_duration_s": 0.05,
            "parallel_response_send_fraction": 0.01,
            "parallel_response_build_duration_s": 0.02,
            "parallel_response_build_fraction": 0.004,
            "parallel_response_pipe_send_duration_s": 0.03,
            "parallel_response_pipe_send_fraction": 0.006,
            "parallel_response_probability_values": 128,
            "parallel_response_probability_values_per_request": 32,
            "parallel_compact_responses": 4,
            "parallel_compact_response_fraction": 1.0,
            "gpu_inference_requests": 4,
            "gpu_inference_batches": 2,
            "gpu_inference_positions": 24,
            "gpu_inference_duration_s": 0.2,
            "gpu_inference_positions_per_batch": 12,
            "gpu_inference_positions_per_request": 6,
            "gpu_inference_batches_per_request": 0.5,
            "gpu_inference_positions_per_second": 120,
        },
        {
            "event": "train_step",
            "preset": "p",
            "loss": 1.0,
            "policy_loss": 0.8,
            "value_loss": 0.2,
            "entropy": 3.0,
            "train_duration_s": 0.5,
        },
        {
            "event": "checkpoint_saved",
            "preset": "p",
            "checkpoint_id": "c1",
            "games_trained": 12,
            "checkpoint_duration_s": 0.25,
            "replay_save_duration_s": 0.75,
            "eval_duration_s": 1.5,
        },
        {
            "event": "evaluation",
            "preset": "p",
            "checkpoint_id": "c1",
            "elo": 900,
            "duration_s": 2.5,
            "evaluation": {
                "opponents": {
                    "heuristic": {"score": 0.25, "failures": 0},
                    "previous_best": {"score": 0.125, "failures": 1},
                }
            },
            "promotion": {"promoted": False},
        },
    ]

    summary = summarize_run(run, loss_window=1)

    assert summary["final_checkpoint_id"] == "c1"
    assert summary["self_play_moves_per_second"] == 2.0
    assert summary["search_moves_per_second"] == 1.6
    assert summary["self_play_policy_targets"]["samples"] == 10
    assert summary["self_play_policy_targets"]["max_prob_mean"]["mean"] == 0.12
    assert summary["self_play_policy_targets"]["diffuse_fraction"]["mean"] == 0.6
    assert summary["self_play_policy_targets"]["transform_active"]["mean"] == 1.0
    assert summary["self_play_policy_targets"]["transform_retained_mass"]["mean"] == 0.25
    assert summary["self_play_policy_targets"]["transform_support_kept_fraction"]["mean"] == 0.15
    assert summary["self_play_policy_targets"]["transform_changed_top1_fraction"]["mean"] == 0.0
    assert summary["self_play_policy_targets"]["transform_max_prob_delta"]["mean"] == 0.06
    assert summary["self_play_policy_targets"]["transform_normalized_entropy_delta"]["mean"] == -0.36
    assert summary["self_play_policy_targets"]["value_draw_fraction"]["mean"] == 0.0
    assert summary["parallel_workers_median"] == 2
    assert summary["parallel_wait_seconds"] == 3.0
    assert summary["parallel_coordination"]["wait_seconds"] == 3.0
    assert summary["parallel_coordination"]["wait_calls"] == 4
    assert summary["parallel_coordination"]["ready_events"] == 2
    assert summary["parallel_coordination"]["ready_pipes"] == 5
    assert summary["parallel_coordination"]["ready_pipes_per_event_median"] == 2.5
    assert summary["parallel_coordination"]["messages"] == 6
    assert summary["parallel_coordination"]["predict_messages"] == 4
    assert summary["parallel_coordination"]["game_result_messages"] == 2
    assert summary["parallel_coordination"]["coalesce_seconds"] == 0.25
    assert summary["parallel_coordination"]["coalesce_calls"] == 2
    assert summary["parallel_coordination"]["coalesce_wait_calls"] == 3
    assert summary["parallel_coordination"]["coalesce_extra_pipes"] == 3
    assert summary["parallel_coordination"]["coalesce_empty_waits"] == 1
    assert summary["parallel_coordination"]["coalesce_extra_pipes_per_call_median"] == 1.5
    assert summary["parallel_coordination"]["coalesce_empty_wait_fraction_median"] == 1 / 3
    assert summary["parallel_coordination"]["payload_build_seconds"] == 0.1
    assert summary["parallel_coordination"]["request_state_bytes_per_position_median"] == 4 * 16 * 16
    assert summary["parallel_coordination"]["request_available_values"] == 512
    assert summary["parallel_coordination"]["compact_requests"] == 4
    assert summary["parallel_coordination"]["compact_request_fraction_median"] == 1.0
    assert summary["parallel_coordination"]["response_send_seconds"] == 0.05
    assert summary["parallel_coordination"]["response_build_seconds"] == 0.02
    assert summary["parallel_coordination"]["response_pipe_send_seconds"] == 0.03
    assert summary["parallel_coordination"]["response_probability_values"] == 128
    assert summary["parallel_coordination"]["response_probability_values_per_request_median"] == 32
    assert summary["parallel_coordination"]["compact_responses"] == 4
    assert summary["parallel_coordination"]["compact_response_fraction_median"] == 1.0
    assert summary["gpu_inference"]["requests"] == 4
    assert summary["gpu_inference"]["batches"] == 2
    assert summary["gpu_inference"]["positions"] == 24
    assert summary["gpu_inference"]["duration_s"] == 0.2
    assert summary["gpu_inference"]["positions_per_second"] == 120
    assert summary["gpu_positions_per_request_median"] == 6
    assert summary["gpu_batches_per_request_median"] == 0.5
    assert summary["gpu_positions_per_batch_median"] == 12
    assert summary["runtime_seconds"]["stream_elapsed"] == 5.0
    assert summary["runtime_seconds"]["parallel_coalesce"] == 0.25
    assert summary["runtime_seconds"]["parallel_payload_build"] == 0.1
    assert summary["runtime_seconds"]["parallel_response_send"] == 0.05
    assert summary["runtime_seconds"]["parallel_response_build"] == 0.02
    assert summary["runtime_seconds"]["parallel_response_pipe_send"] == 0.03
    assert summary["runtime_seconds"]["external_eval"] == 2.5
    assert summary["loss_first_last"]["loss"]["first_mean"] == 1.0
    assert summary["final_external_eval"]["heuristic_score"] == 0.25
    assert summary["final_external_eval"]["previous_best_score"] == 0.125
    assert summary["final_external_eval"]["failures"] == 1


def test_audit_training_log_selects_checkpoint_run(tmp_path):
    log_path = tmp_path / "training_log.jsonl"
    log_path.write_text(
        "\n".join([
            '{"event":"run_start","preset":"p","timestamp":"first"}',
            '{"event":"checkpoint_saved","preset":"p","checkpoint_id":"old"}',
            '{"event":"run_start","preset":"p","timestamp":"second"}',
            (
                '{"event":"self_play_game","preset":"p","moves":4,'
                '"search_moves":3,"parallel_stream_elapsed_s":2.0}'
            ),
            '{"event":"checkpoint_saved","preset":"p","checkpoint_id":"new"}',
        ])
        + "\n",
        encoding="utf-8",
    )

    summary = audit_training_log(log_path, preset="p", checkpoint_id="new")

    assert summary["run_start"] == "second"
    assert summary["final_checkpoint_id"] == "new"
    assert summary["self_play_moves_per_second"] == 2.0


def test_short_training_run_reports_adaptive_loss_edges():
    run = [{"event": "run_start", "timestamp": "start", "preset": "p"}]
    for value in (6.0, 5.0, 4.0):
        run.append({
            "event": "train_step",
            "preset": "p",
            "loss": value,
            "policy_loss": value - 1.0,
            "value_loss": 1.0,
            "entropy": 5.0,
        })

    summary = summarize_run(run, loss_window=32)

    loss = summary["loss_first_last"]["loss"]
    assert loss["samples"] == 3
    assert loss["requested_window"] == 32
    assert loss["window"] == 1
    assert loss["first_mean"] == 6.0
    assert loss["last_mean"] == 4.0
    assert loss["delta"] == -2.0
    assert loss_delta(summary, "loss") == -2.0


def test_attach_replay_quality_adds_replay_assessment(tmp_path):
    replay_path = tmp_path / "replay.pkl"
    samples = [
        (
            np.zeros((4, 2, 2), dtype=np.float32),
            np.full(100, 0.01, dtype=np.float64),
            1.0,
        ),
        (
            np.zeros((4, 2, 2), dtype=np.float32),
            np.full(100, 0.01, dtype=np.float64),
            -1.0,
        ),
    ]
    with replay_path.open("wb") as handle:
        pickle.dump({"version": 1, "games_recorded": 2, "samples": samples}, handle)

    summary = attach_replay_quality({}, replay_path=replay_path, replay_max_samples=0)

    assert summary["replay_quality"]["audited_samples"] == 2
    assert (
        summary["replay_quality"]["replay_quality_assessment"]["label"]
        == "diffuse_policy_targets"
    )


def test_parse_gpu_smi_log_filters_and_summarizes_run_window(tmp_path):
    gpu_log = tmp_path / "gpu_smi.log"
    gpu_log.write_text(
        """
===== 2026-06-05T22:13:00+00:00 =====
|  0%   40C    P8            N/A  /  115W |       0MiB /   8188MiB |      0%      Default |
===== 2026-06-05T22:14:20+00:00 =====
|  0%   46C    P2            N/A  /  115W |    1236MiB /   8188MiB |      0%      Default |
===== 2026-06-05T22:15:20+00:00 =====
|  0%   51C    P2            N/A  /  115W |    1450MiB /   8188MiB |     71%      Default |
===== 2026-06-06T03:21:00+00:00 =====
|  0%   46C    P8            N/A  /  115W |       0MiB /   8188MiB |      0%      Default |
""".strip()
        + "\n",
        encoding="utf-8",
    )

    samples = parse_gpu_smi_log(
        gpu_log,
        start="2026-06-05T22:14:17+00:00",
        end="2026-06-06T03:20:36+00:00",
    )
    summary = summarize_gpu_smi_samples(samples)

    assert [sample["gpu_util_percent"] for sample in samples] == [0, 71]
    assert summary["samples"] == 2
    assert summary["gpu_util_percent_max"] == 71
    assert summary["gpu_util_nonzero_fraction"] == 0.5
    assert summary["gpu_util_idle_fraction"] == 0.5
    assert summary["memory_used_mb_max"] == 1450


def test_resource_monitor_samples_filter_and_summarize(tmp_path):
    resource_log = tmp_path / "resource_monitor.jsonl"
    resource_log.write_text(
        "\n".join([
            (
                '{"timestamp":"2026-06-05T22:13:00+00:00",'
                '"cpu_util_percent":10.0,"load":{"load1":1.0},'
                '"memory":{"memory_used_percent":20.0},'
                '"gpu":{"gpu_util_percent":0}}'
            ),
            (
                '{"timestamp":"2026-06-05T22:14:20+00:00",'
                '"cpu_util_percent":50.0,"cpu_used_cores":6.0,'
                '"host_cpu_util_percent":95.0,'
                '"cpu_throttled_period_fraction":0.1,'
                '"load":{"load1":4.0},'
                '"memory":{"memory_used_percent":30.0},'
                '"gpu":{"gpu_util_percent":0}}'
            ),
            (
                '{"timestamp":"2026-06-05T22:15:20+00:00",'
                '"cpu_util_percent":90.0,"cpu_used_cores":10.0,'
                '"host_cpu_util_percent":98.0,'
                '"cpu_throttled_period_fraction":0.3,'
                '"load":{"load1":8.0},'
                '"memory":{"memory_used_percent":40.0},'
                '"gpu":{"gpu_util_percent":70}}'
            ),
        ])
        + "\n",
        encoding="utf-8",
    )

    samples = load_resource_monitor_samples(
        resource_log,
        start="2026-06-05T22:14:17+00:00",
        end="2026-06-05T22:16:00+00:00",
    )
    summary = summarize_resource_monitor_samples(samples)

    assert len(samples) == 2
    assert summary["cpu_util_percent"]["mean"] == 70.0
    assert summary["cpu_util_percent"]["max"] == 90.0
    assert summary["cpu_used_cores"]["mean"] == 8.0
    assert summary["host_cpu_util_percent"]["mean"] == 96.5
    assert summary["cpu_throttled_period_fraction"]["max"] == 0.3
    assert summary["cpu_util_high_fraction"] == 0.5
    assert summary["load1"]["median"] == 6.0
    assert summary["memory_used_percent"]["max"] == 40.0
    assert summary["gpu_util_idle_fraction"] == 0.5


def test_classify_bottleneck_detects_mcts_search_coordination_bound():
    summary = {
        "self_play_moves_per_second": 3.2,
        "gpu_inference": {
            "positions_per_second": 2100.0,
            "duration_s": 3600.0,
        },
        "runtime_seconds": {"stream_elapsed": 10800.0},
        "gpu_smi": {
            "gpu_util_idle_fraction": 0.83,
            "gpu_util_percent_mean": 4.0,
            "gpu_util_percent_median": 0.0,
            "gpu_util_percent_max": 91,
        },
    }

    assessment = classify_bottleneck(summary)

    assert assessment["label"] == "mcts_search_coordination_bound"
    assert assessment["confidence"] == "high"


def test_classify_bottleneck_detects_cpu_bound_with_resource_samples():
    summary = {
        "self_play_moves_per_second": 5.0,
        "gpu_smi": {"gpu_util_idle_fraction": 0.75},
        "resources": {
            "cpu_util_percent": {"mean": 88.0, "median": 90.0, "max": 98.0},
            "cpu_util_high_fraction": 0.75,
        },
    }

    assessment = classify_bottleneck(summary)

    assert assessment["label"] == "cpu_bound"
    assert assessment["confidence"] == "medium"


def test_classify_bottleneck_detects_gpu_bound_when_utilization_is_sustained():
    summary = {
        "self_play_moves_per_second": 20.0,
        "gpu_smi": {
            "gpu_util_idle_fraction": 0.05,
            "gpu_util_percent_mean": 85.0,
            "gpu_util_percent_median": 88.0,
            "gpu_util_percent_max": 99.0,
        },
    }

    assessment = classify_bottleneck(summary)

    assert assessment["label"] == "gpu_bound"
    assert assessment["confidence"] == "medium"


def test_recommend_next_action_combines_mcts_bottleneck_and_diffuse_replay():
    summary = {
        "bottleneck_assessment": {"label": "mcts_search_coordination_bound"},
        "replay_quality": {
            "replay_quality_assessment": {"label": "diffuse_policy_targets"},
        },
        "loss_first_last": {
            "loss": {"first_mean": 5.0, "last_mean": 4.5},
            "policy_loss": {"first_mean": 4.0, "last_mean": 3.8},
            "value_loss": {"first_mean": 1.0, "last_mean": 0.8},
            "entropy": {"first_mean": 5.0, "last_mean": 4.9},
        },
    }

    recommendation = recommend_next_action(summary)

    assert recommendation["label"] == "fix_search_target_alignment_before_scaling"
    assert recommendation["loss_improving"] is True
    assert "diffuse" in recommendation["recommendation"]


def test_recommend_next_action_continues_when_eval_improves():
    summary = {
        "bottleneck_assessment": {"label": "undetermined"},
        "final_external_eval": {
            "heuristic_score": 0.6,
            "previous_best_score": 0.6,
            "promoted": False,
        },
    }

    recommendation = recommend_next_action(summary)

    assert recommendation["label"] == "continue_same_recipe_longer"


def test_recommend_next_action_flags_draw_heavy_value_targets():
    summary = {
        "bottleneck_assessment": {"label": "undetermined"},
        "replay_quality": {
            "replay_quality_assessment": {"label": "draw_heavy_value_targets"},
        },
        "loss_first_last": {
            "loss": {"first_mean": 5.0, "last_mean": 4.5},
            "policy_loss": {"first_mean": 4.0, "last_mean": 3.8},
            "value_loss": {"first_mean": 1.0, "last_mean": 0.8},
        },
    }

    recommendation = recommend_next_action(summary)

    assert recommendation["label"] == "revise_value_targets_draw_handling"
