import improve


def test_improvement_loop_passes_eval_diagnostic_options(monkeypatch):
    evaluate_calls = []
    training_calls = []

    def fake_run_baseline_training(**kwargs):
        training_calls.append(kwargs)
        return {"final": {"id": "candidate_final"}}

    monkeypatch.setattr(improve, "run_baseline_training", fake_run_baseline_training)

    def fake_evaluate_checkpoint(agent_id, **kwargs):
        evaluate_calls.append({"agent_id": agent_id, **kwargs})
        return {
            "evaluation": {"elo": 1000},
            "promotion": {"gate_passed": False, "promoted": False},
        }

    monkeypatch.setattr(improve, "evaluate_checkpoint", fake_evaluate_checkpoint)

    improve.run_improvement_loop(
        presets=("large_16x16_top_human_gpu",),
        checkpoint_dir="checkpoints",
        eval_games=4,
        previous_best_games=2,
        max_runtime_minutes=3,
        max_runtime_dispatch_margin_minutes=1.5,
        eval_n_playout=64,
        opponent_eval_n_playout=32,
        eval_mode="tactical_beam",
        opponent_eval_mode="native",
        stop_on_gate=False,
    )

    assert training_calls == [{
        "preset": "large_16x16_top_human_gpu",
        "checkpoint_dir": "checkpoints",
        "resume_best": True,
        "max_runtime_minutes": 3,
        "max_runtime_dispatch_margin_minutes": 1.5,
    }]
    assert evaluate_calls == [{
        "agent_id": "candidate_final",
        "registry_path": "checkpoints/registry.json",
        "games": 4,
        "previous_best_games": 2,
        "n_playout": 64,
        "opponent_n_playout": 32,
        "eval_mode": "tactical_beam",
        "opponent_eval_mode": "native",
        "promote": True,
    }]
