import evaluator


def _agent(agent_id, *, preset="eval_test", n_playout=128, elo=1000, games_trained=28):
    return {
        "id": agent_id,
        "name": agent_id,
        "type": "checkpoint",
        "preset": preset,
        "elo": elo,
        "games_trained": games_trained,
        "n_playout": n_playout,
        "board_width": 16,
        "board_height": 16,
        "n_in_row": 5,
        "num_res_blocks": 4,
        "num_filters": 64,
        "metrics": {
            "config": {
                "eval_parallel_games": "auto",
                "eval_parallel_cap_to_cpu": True,
                "eval_parallel_cpu_multiplier": 1.0,
            }
        },
    }


def test_evaluate_checkpoint_logs_opponent_progress_and_uses_playout_overrides(monkeypatch):
    candidate = _agent("candidate")
    champion = _agent("champion", elo=1134, n_playout=16)
    events = []
    match_calls = []

    def fake_match_record(
        candidate_arg,
        opponent,
        games,
        candidate_n_playout,
        opponent_n_playout,
        candidate_eval_mode="mcts",
        opponent_eval_mode="mcts",
    ):
        match_calls.append({
            "opponent_id": opponent["id"],
            "games": games,
            "candidate_n_playout": candidate_n_playout,
            "opponent_n_playout": opponent_n_playout,
            "candidate_eval_mode": candidate_eval_mode,
            "opponent_eval_mode": opponent_eval_mode,
        })
        wins_by_opponent = {
            "random": games,
            "heuristic": 1,
            "champion": 2,
        }
        wins = wins_by_opponent[opponent["id"]]
        losses = games - wins
        return {
            "opponent_id": opponent["id"],
            "opponent_name": opponent.get("name", opponent["id"]),
            "opponent_elo": opponent.get("elo", 1000),
            "wins": wins,
            "draws": 0,
            "losses": losses,
            "games": games,
            "score": wins / games,
            "failures": 0,
            "failure_messages": [],
            "duration_s": 1.5,
            "parallel_workers": min(4, games),
        }

    monkeypatch.setattr(evaluator, "find_agent", lambda *_args, **_kwargs: candidate)
    monkeypatch.setattr(
        evaluator,
        "champion_checkpoint",
        lambda *_args, **_kwargs: champion,
    )
    monkeypatch.setattr(evaluator, "_match_record", fake_match_record)
    monkeypatch.setattr(evaluator, "_eval_parallel_workers", lambda _candidate, games: min(4, games))
    monkeypatch.setattr(
        evaluator,
        "append_training_event",
        lambda event, checkpoint_dir="checkpoints": events.append((checkpoint_dir, event)),
    )

    result = evaluator.evaluate_checkpoint(
        "candidate",
        registry_path="tmp_checkpoints/registry.json",
        games=4,
        previous_best_games=2,
        n_playout=64,
        opponent_n_playout=32,
        eval_mode="tactical_beam",
        opponent_eval_mode="native",
        promote=False,
    )

    event_names = [event["event"] for _checkpoint_dir, event in events]
    assert event_names == [
        "evaluation_opponent_start",
        "evaluation_opponent",
        "evaluation_opponent_start",
        "evaluation_opponent",
        "evaluation_opponent_start",
        "evaluation_opponent",
        "evaluation",
    ]
    opponent_keys = [
        event["opponent_key"]
        for _checkpoint_dir, event in events
        if event["event"] == "evaluation_opponent"
    ]
    assert opponent_keys == ["random", "heuristic", "previous_best"]
    assert events[0][0].as_posix() == "tmp_checkpoints"
    for _checkpoint_dir, event in events[:-1]:
        assert event["checkpoint_id"] == "candidate"
        assert event["games_trained"] == 28
        assert event["n_playout"] == 64
        assert event["parallel_workers"] in {2, 4}
    assert events[1][1]["score"] == 1.0
    assert events[3][1]["score"] == 0.25
    assert events[5][1]["games"] == 2
    assert events[5][1]["score"] == 1.0
    assert events[-1][1]["games_trained"] == 28
    assert result["evaluation"]["n_playout"] == 64
    assert result["evaluation"]["opponent_n_playout"] == {
        "baseline": 0,
        "previous_best": 32,
        "checkpoint_override": 32,
    }
    assert match_calls == [
        {
            "opponent_id": "random",
            "games": 4,
            "candidate_n_playout": 64,
            "opponent_n_playout": 0,
            "candidate_eval_mode": "tactical_beam",
            "opponent_eval_mode": "random",
        },
        {
            "opponent_id": "heuristic",
            "games": 4,
            "candidate_n_playout": 64,
            "opponent_n_playout": 0,
            "candidate_eval_mode": "tactical_beam",
            "opponent_eval_mode": "heuristic",
        },
        {
            "opponent_id": "champion",
            "games": 2,
            "candidate_n_playout": 64,
            "opponent_n_playout": 32,
            "candidate_eval_mode": "tactical_beam",
            "opponent_eval_mode": "native",
        },
    ]


def test_match_record_aggregates_move_counts_by_outcome(monkeypatch):
    candidate = _agent("candidate")
    opponent = {
        "id": "heuristic",
        "type": "baseline",
        "name": "One-ply heuristic",
        "elo": 1000,
        "board_width": 16,
        "board_height": 16,
        "n_in_row": 5,
    }

    monkeypatch.setattr(evaluator, "_eval_parallel_workers", lambda _candidate, games: 1)
    monkeypatch.setattr(
        evaluator,
        "_match_chunk",
        lambda _args: [
            {"game": 1, "outcome": "win", "moves": 41},
            {"game": 2, "outcome": "loss", "moves": 53},
            {"game": 3, "outcome": "draw", "moves": 256},
            {"game": 4, "outcome": "failure", "error": "boom"},
        ],
    )

    record = evaluator._match_record(
        candidate,
        opponent,
        games=4,
        candidate_n_playout=64,
        opponent_n_playout=0,
    )

    assert record["wins"] == 1
    assert record["losses"] == 1
    assert record["draws"] == 1
    assert record["failures"] == 1
    assert record["avg_moves"] == 116.667
    assert record["win_avg_moves"] == 41
    assert record["loss_avg_moves"] == 53
    assert record["draw_avg_moves"] == 256
