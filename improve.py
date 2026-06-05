from config import IMPROVEMENT_PRESETS
from evaluator import evaluate_checkpoint
from train import run_baseline_training


def run_improvement_loop(
    presets=None,
    eval_games=4,
    previous_best_games=None,
    checkpoint_dir="checkpoints",
    resume_best=True,
    rounds=1,
    stop_on_gate=True,
):
    presets = tuple(presets or IMPROVEMENT_PRESETS)
    rounds = max(1, int(rounds))
    results = []
    for round_idx in range(1, rounds + 1):
        for preset in presets:
            training_result = run_baseline_training(
                preset=preset,
                checkpoint_dir=checkpoint_dir,
                resume_best=resume_best,
            )
            final = training_result["final"]
            evaluation_result = evaluate_checkpoint(
                final["id"],
                registry_path=f"{checkpoint_dir}/registry.json",
                games=eval_games,
                previous_best_games=previous_best_games or eval_games,
                promote=True,
            )
            result = {
                "round": round_idx,
                "preset": preset,
                "checkpoint": final,
                "evaluation": evaluation_result,
            }
            results.append(result)
            if stop_on_gate and evaluation_result["promotion"]["gate_passed"]:
                return results
    return results
