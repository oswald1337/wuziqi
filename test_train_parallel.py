import logging
import numpy as np

from config import get_training_preset
from game import Board
from model import PolicyValueNet
from train import (
    _RemotePolicyValueNet,
    _RuntimeBudget,
    _decode_request_availables,
    _evaluate_remote_policy_payloads,
    _evaluate_policy,
    _public_config,
    _parallel_self_play_worker_count,
    _serve_parallel_self_play_batch,
    _serve_parallel_self_play_stream,
    _start_parallel_self_play_workers,
    _stop_parallel_self_play_workers,
)


def test_parallel_self_play_batch_uses_brokered_inference():
    config = get_training_preset("debug")
    config.update({
        "self_play_parallel_games": 2,
        "n_playout": 2,
        "mcts_batch_size": 2,
        "gpu_inference_max_batch_size": 4,
        "mcts_heuristic_prior_weight": 0.0,
        "use_gpu": False,
    })
    policy_value_net = PolicyValueNet(
        config["board_width"],
        config["board_height"],
        use_gpu=False,
        num_res_blocks=config["num_res_blocks"],
        num_filters=config["num_filters"],
        architecture=config.get("architecture", "residual"),
    )
    logger = logging.getLogger("test_parallel_self_play")
    workers, pipes = _start_parallel_self_play_workers(config, logger)
    try:
        results, metrics = _serve_parallel_self_play_batch(
            policy_value_net,
            config,
            seed=123,
            game_indices=[1, 2],
            workers=workers,
            pipes=pipes,
            logger=logger,
        )
    finally:
        _stop_parallel_self_play_workers(workers, pipes, logger)

    assert set(results) == {1, 2}
    assert metrics["parallel_workers"] == 2
    assert metrics["parallel_batch_games"] == 2
    assert metrics["gpu_inference_requests"] > 0
    assert metrics["gpu_inference_batches"] > 0
    assert metrics["gpu_inference_positions"] > 0
    assert metrics["gpu_inference_positions_per_request"] > 0
    assert metrics["parallel_wait_calls"] > 0
    assert metrics["parallel_ready_events"] > 0
    assert metrics["parallel_ready_pipes"] >= metrics["parallel_ready_events"]
    assert metrics["parallel_messages"] >= metrics["gpu_inference_requests"]
    assert "parallel_coalesce_calls" in metrics
    assert "parallel_coalesce_empty_wait_fraction" in metrics or metrics["parallel_coalesce_wait_calls"] == 0
    assert "parallel_payload_build_duration_s" in metrics
    assert metrics["parallel_compact_requests"] == metrics["gpu_inference_requests"]
    assert metrics["parallel_compact_request_fraction"] == 1.0
    assert metrics["parallel_request_state_bytes"] > 0
    assert metrics["parallel_request_state_bytes_per_position"] == (
        4 * config["board_width"] * config["board_height"]
    )
    assert metrics["parallel_request_available_values"] > 0
    assert "parallel_response_send_duration_s" in metrics
    assert metrics["parallel_compact_responses"] == metrics["gpu_inference_requests"]
    assert metrics["parallel_compact_response_fraction"] == 1.0
    assert metrics["parallel_response_probability_values"] > 0
    for result in results.values():
        assert result["type"] == "game_result"
        assert result["moves"]
        assert result["play_data"]


def test_parallel_self_play_stream_processes_games_incrementally():
    config = get_training_preset("debug")
    config.update({
        "self_play_parallel_games": 2,
        "n_playout": 2,
        "mcts_batch_size": 2,
        "gpu_inference_max_batch_size": 4,
        "mcts_heuristic_prior_weight": 0.0,
        "parallel_metrics_games": 1,
        "parallel_metrics_interval_s": 1,
        "use_gpu": False,
    })
    policy_value_net = PolicyValueNet(
        config["board_width"],
        config["board_height"],
        use_gpu=False,
        num_res_blocks=config["num_res_blocks"],
        num_filters=config["num_filters"],
        architecture=config.get("architecture", "residual"),
    )
    logger = logging.getLogger("test_parallel_self_play_stream")
    workers, pipes = _start_parallel_self_play_workers(config, logger)
    results = {}
    metric_events = []
    try:
        stream_summary = _serve_parallel_self_play_stream(
            policy_value_net,
            config,
            seed=456,
            game_indices=[1, 2, 3],
            workers=workers,
            pipes=pipes,
            logger=logger,
            on_game_result=lambda game_idx, result, metrics: results.__setitem__(
                game_idx,
                (result, metrics),
            ),
            on_metrics=metric_events.append,
        )
    finally:
        _stop_parallel_self_play_workers(workers, pipes, logger)

    assert set(results) == {1, 2, 3}
    assert stream_summary["completed_games"] == 3
    assert stream_summary["remaining_games"] == 0
    assert stream_summary["stopped_early"] is False
    assert metric_events
    assert sum(event["parallel_batch_games"] for event in metric_events) == 3
    assert any(event["gpu_inference_positions"] > 0 for event in metric_events)
    assert any(
        event.get("gpu_inference_positions_per_request", 0) > 0
        for event in metric_events
    )
    assert any(event.get("parallel_wait_calls", 0) > 0 for event in metric_events)
    assert any(event.get("parallel_compact_request_fraction") == 1.0 for event in metric_events)
    assert any(event.get("parallel_compact_response_fraction") == 1.0 for event in metric_events)
    for result, metrics in results.values():
        assert result["type"] == "game_result"
        assert result["moves"]
        assert result["play_data"]
        assert metrics["parallel_workers"] == 2


def test_parallel_self_play_stream_stops_dispatching_on_budget():
    config = get_training_preset("debug")
    config.update({
        "self_play_parallel_games": 2,
        "n_playout": 2,
        "mcts_batch_size": 2,
        "gpu_inference_max_batch_size": 4,
        "mcts_heuristic_prior_weight": 0.0,
        "parallel_metrics_games": 1,
        "parallel_metrics_interval_s": 1,
        "use_gpu": False,
    })
    policy_value_net = PolicyValueNet(
        config["board_width"],
        config["board_height"],
        use_gpu=False,
        num_res_blocks=config["num_res_blocks"],
        num_filters=config["num_filters"],
        architecture=config.get("architecture", "residual"),
    )
    logger = logging.getLogger("test_parallel_self_play_stream_budget")
    workers, pipes = _start_parallel_self_play_workers(config, logger)
    results = {}
    metric_events = []

    def should_stop(_estimated_game_duration_s):
        return bool(results)

    try:
        stream_summary = _serve_parallel_self_play_stream(
            policy_value_net,
            config,
            seed=789,
            game_indices=[1, 2, 3, 4],
            workers=workers,
            pipes=pipes,
            logger=logger,
            on_game_result=lambda game_idx, result, metrics: results.__setitem__(
                game_idx,
                (result, metrics),
            ),
            on_metrics=metric_events.append,
            should_stop=should_stop,
        )
    finally:
        _stop_parallel_self_play_workers(workers, pipes, logger)

    assert len(results) == 2
    assert set(results).issubset({1, 2})
    assert stream_summary["requested_games"] == 4
    assert stream_summary["dispatched_games"] == 2
    assert stream_summary["completed_games"] == 2
    assert stream_summary["remaining_games"] == 2
    assert stream_summary["stopped_early"] is True
    assert stream_summary["dispatch_stop_estimated_game_s"] > 0
    assert metric_events


def test_runtime_budget_zero_expires_immediately():
    budget = _RuntimeBudget(0)

    assert budget.enabled is True
    assert budget.expired() is True
    assert budget.should_stop_dispatch(estimated_unit_s=0) is True
    summary = budget.summary()
    assert summary["max_runtime_s"] == 0
    assert summary["runtime_budget_exceeded"] is True
    assert summary["runtime_dispatch_stopped"] is True


def test_runtime_budget_stops_dispatch_before_projected_overrun():
    budget = _RuntimeBudget(10, dispatch_margin_minutes=1)
    budget.start_s -= 520

    assert budget.expired() is False
    assert budget.should_stop_dispatch(estimated_unit_s=30) is True
    summary = budget.summary()
    assert summary["runtime_budget_exceeded"] is False
    assert summary["runtime_dispatch_stopped"] is True
    assert summary["runtime_dispatch_margin_s"] == 60


def test_remote_policy_sends_compact_uint8_state_requests():
    class FakeConn:
        def __init__(self):
            self.payload = None

        def send(self, payload):
            self.payload = payload

        def recv(self):
            evaluations = []
            for _state in self.payload["states"]:
                evaluations.append({
                    "actions": [0],
                    "probs": [1.0],
                    "value": 0.0,
                })
            return {
                "type": "prediction_batch",
                "request_id": self.payload["request_id"],
                "evaluations": evaluations,
            }

    board_a = Board(width=3, height=3, n_in_row=3)
    board_b = Board(width=3, height=3, n_in_row=3)
    board_a.init_board()
    board_b.init_board()
    board_b.do_move(0)

    conn = FakeConn()
    policy = _RemotePolicyValueNet(conn)

    policy.policy_value_batch_fn([board_a, board_b])

    assert isinstance(conn.payload["states"], np.ndarray)
    assert conn.payload["states"].dtype == np.uint8
    assert conn.payload["states"].shape == (2, 4, 3, 3)
    assert conn.payload["states"].flags.c_contiguous
    assert conn.payload["request_format"] == "compact_v1"
    assert conn.payload["available_lengths"].tolist() == [
        len(board_a.availables),
        len(board_b.availables),
    ]
    assert conn.payload["flat_availables"].dtype == np.uint16
    assert conn.payload["response_format"] == "compact_v1"


def test_remote_policy_decodes_compact_response_with_local_legal_order():
    class FakeConn:
        def __init__(self):
            self.payload = None

        def send(self, payload):
            self.payload = payload

        def recv(self):
            lengths = np.asarray(
                self.payload["available_lengths"],
                dtype=np.int32,
            )
            flat_probs = np.concatenate([
                np.full(length, 1.0 / length, dtype=np.float32)
                for length in lengths
            ])
            return {
                "type": "prediction_batch",
                "request_id": self.payload["request_id"],
                "response_format": "compact_v1",
                "prob_lengths": lengths,
                "flat_probs": flat_probs,
                "values": np.asarray([0.25, -0.5], dtype=np.float32),
            }

    board_a = Board(width=3, height=3, n_in_row=3)
    board_b = Board(width=3, height=3, n_in_row=3)
    board_a.init_board()
    board_b.init_board()
    board_b.do_move(0)

    policy = _RemotePolicyValueNet(FakeConn())
    results = policy.policy_value_batch_fn([board_a, board_b])

    assert results[0][0][0][0] == board_a.availables[0]
    assert results[1][0][0][0] == board_b.availables[0]
    assert results[0][1] == 0.25
    assert results[1][1] == -0.5


def test_remote_policy_payload_evaluation_keeps_compact_probs_as_arrays():
    class FakePolicy:
        def __init__(self):
            self.batch_shapes = []

        def policy_value(self, states):
            self.batch_shapes.append(states.shape)
            probs = np.full((len(states), 9), 1.0 / 9.0, dtype=np.float32)
            values = np.arange(len(states), dtype=np.float32).reshape(-1, 1)
            return probs, values

    payloads = [
        {
            "states": np.zeros((2, 4, 3, 3), dtype=np.float32),
            "availables_batch": [[0, 1, 2], [4, 5]],
            "response_format": "compact_v1",
        },
        {
            "states": np.ones((1, 4, 3, 3), dtype=np.float32),
            "availables_batch": [[6, 7, 8]],
            "response_format": "compact_v1",
        },
    ]
    policy = FakePolicy()

    evaluations, batches = _evaluate_remote_policy_payloads(
        policy,
        payloads,
        {"gpu_inference_max_batch_size": 8},
    )

    assert batches == 1
    assert policy.batch_shapes == [(3, 4, 3, 3)]
    assert len(evaluations) == 3
    assert "actions" not in evaluations[0]
    assert isinstance(evaluations[0]["probs"], np.ndarray)
    assert evaluations[0]["probs"].dtype == np.float32
    assert np.isclose(float(np.sum(evaluations[0]["probs"])), 1.0)


def test_decode_compact_request_availables_uses_flat_uint16_payload():
    message = {
        "request_format": "compact_v1",
        "available_lengths": np.asarray([3, 2], dtype=np.uint16),
        "flat_availables": np.asarray([0, 1, 2, 4, 5], dtype=np.uint16),
    }

    availables, values = _decode_request_availables(message, state_count=2)

    assert values == 5
    assert [item.tolist() for item in availables] == [[0, 1, 2], [4, 5]]


def test_public_config_preserves_parallel_efficiency_knobs():
    config = get_training_preset("large_16x16_top_human_gpu")
    public = _public_config(config)

    assert public["self_play_parallel_games"] == "auto"
    assert public["self_play_parallel_cap_to_cpu"] is True
    assert public["self_play_parallel_cpu_multiplier"] == 1.0
    assert public["eval_parallel_games"] == "auto"
    assert public["eval_parallel_cap_to_cpu"] is True
    assert public["eval_parallel_cpu_multiplier"] == 1.0
    assert public["internal_eval_games"] == 0
    assert public["mcts_batch_size"] == 128
    assert public["mcts_min_batches_per_search"] == 4
    assert public["gpu_inference_max_batch_size"] == 512
    assert public["gpu_inference_coalesce_ms"] == 25
    assert public["gpu_inference_coalesce_slice_ms"] == 3
    assert public["gpu_inference_compact_request"] is True
    assert public["gpu_inference_state_dtype"] == "uint8"
    assert public["gpu_inference_compact_response"] is True
    assert public["self_play_target_transform"] == "top_k"
    assert public["self_play_target_top_k"] == 16


def test_parallel_self_play_worker_count_uses_cpu_quota_for_auto(monkeypatch):
    import train

    monkeypatch.setattr(train, "usable_cpu_count", lambda multiplier=1.0: 12)

    assert _parallel_self_play_worker_count({"self_play_parallel_games": "auto"}) == 12


def test_internal_eval_can_be_disabled_without_calling_policy():
    class FailingPolicy:
        def policy_value_fn(self, *_args, **_kwargs):
            raise AssertionError("internal eval should be skipped")

        def policy_value_batch_fn(self, *_args, **_kwargs):
            raise AssertionError("internal eval should be skipped")

    config = get_training_preset("debug")
    config["internal_eval_games"] = 0

    results, elo = _evaluate_policy(FailingPolicy(), config)

    assert elo == 1000
    assert results["skipped"] is True
    assert results["games"] == 0


def test_parallel_self_play_worker_count_caps_requested_workers(monkeypatch):
    import train

    monkeypatch.setattr(train, "usable_cpu_count", lambda multiplier=1.0: 12)

    assert _parallel_self_play_worker_count({"self_play_parallel_games": 128}) == 12


def test_parallel_self_play_worker_count_can_disable_cpu_cap(monkeypatch):
    import train

    monkeypatch.setattr(train, "usable_cpu_count", lambda multiplier=1.0: 12)

    assert _parallel_self_play_worker_count({
        "self_play_parallel_games": 128,
        "self_play_parallel_cap_to_cpu": False,
    }) == 128


def test_eval_parallel_workers_use_cpu_quota_for_auto(monkeypatch):
    import train
    from evaluator import _eval_parallel_workers

    monkeypatch.setattr(train, "usable_cpu_count", lambda multiplier=1.0: 12)
    candidate = {"metrics": {"config": {"eval_parallel_games": "auto"}}}

    assert _eval_parallel_workers(candidate, games=32) == 12
    assert _eval_parallel_workers(candidate, games=8) == 8
