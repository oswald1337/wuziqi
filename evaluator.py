import copy
import concurrent.futures
import multiprocessing as mp
import time
import traceback
from pathlib import Path

from checkpoint_registry import (
    champion_checkpoint,
    checkpoint_architecture,
    elo_after_games,
    find_agent,
    resolve_model_path,
    score_from_record,
    update_checkpoint,
)
from config import COMPLETION_GATE, PROMOTION_GATE
from game import Board, Game
from mcts import MCTSPlayer
from model import PolicyValueNet
from players import HeuristicPlayer, RandomPlayer
from tracking import append_training_event
from train import TacticalBeamSelfPlayPlayer, _resolve_parallel_worker_count


def _board_config(agent):
    return {
        "board_width": agent["board_width"],
        "board_height": agent["board_height"],
        "n_in_row": agent["n_in_row"],
    }


def _checkpoint_policy(agent):
    return PolicyValueNet(
        agent["board_width"],
        agent["board_height"],
        model_file=str(resolve_model_path(agent)),
        use_gpu=False,
        num_res_blocks=agent.get("num_res_blocks", 4),
        num_filters=agent.get("num_filters", 64),
        architecture=checkpoint_architecture(agent),
    )


def _agent_mcts_tactical_threshold(agent):
    return (
        agent.get("mcts_tactical_threshold")
        or agent.get("metrics", {}).get("config", {}).get("mcts_tactical_threshold")
    )


def _agent_mcts_tactical_prior(agent):
    config = _agent_config(agent)
    return {
        "weight": config.get("mcts_tactical_prior_weight", 0.0),
        "temperature": config.get("mcts_tactical_prior_temperature", 1.0),
        "two_ply_bonus": config.get("mcts_tactical_prior_two_ply_bonus", 0.0),
        "two_ply_max_candidates": config.get("mcts_tactical_prior_two_ply_max_candidates", 16),
        "two_ply_max_replies": config.get("mcts_tactical_prior_two_ply_max_replies", 6),
        "two_ply_max_followups": config.get("mcts_tactical_prior_two_ply_max_followups", 12),
    }


def _agent_mcts_tactical_leaf(agent):
    config = _agent_config(agent)
    return {
        "eval": config.get("mcts_tactical_leaf_eval", False),
        "win_value": config.get("mcts_tactical_leaf_win_value", 1.0),
        "loss_value": config.get("mcts_tactical_leaf_loss_value", 0.95),
        "forcing_value": config.get("mcts_tactical_leaf_forcing_value", 0.85),
        "two_ply_value": config.get("mcts_tactical_leaf_two_ply_value", 0.70),
        "two_ply": config.get("mcts_tactical_leaf_two_ply", False),
        "max_candidates": config.get("mcts_tactical_leaf_max_candidates", 16),
        "max_replies": config.get("mcts_tactical_leaf_max_replies", 6),
        "max_followups": config.get("mcts_tactical_leaf_max_followups", 12),
    }


def _agent_config(agent):
    return agent.get("metrics", {}).get("config", {})


def _agent_eval_mode(agent, eval_mode):
    if eval_mode == "native":
        return _agent_config(agent).get("self_play_mode", "mcts")
    return eval_mode


class _FixedTempPlayer:
    def __init__(self, player, temp=1e-6):
        self._player = player
        self._temp = temp
        self.player = None

    def set_player_ind(self, player):
        self.player = player
        self._player.set_player_ind(player)

    def reset_player(self):
        if hasattr(self._player, "reset_player"):
            self._player.reset_player()

    def get_action(self, board):
        return self._player.get_action(board, temp=self._temp)


def _beam_player(policy, agent, seed=0):
    config = _agent_config(agent)
    return _FixedTempPlayer(
        TacticalBeamSelfPlayPlayer(
            policy,
            tactical_guard=config.get("policy_tactical_guard", True),
            beam_width=config.get("beam_width", 8),
            policy_top_k=config.get("beam_policy_top_k", 8),
            value_weight=config.get("beam_value_weight", 1.2),
            policy_weight=config.get("beam_policy_weight", 0.45),
            tactical_weight=config.get("beam_tactical_weight", 1.0),
            reply_penalty=config.get("beam_reply_penalty", 0.35),
            fork_weight=config.get("beam_fork_weight", 0.0),
            fork_threshold=config.get("beam_fork_threshold", 250000.0),
            dirichlet_alpha=config.get("policy_dirichlet_alpha", 0.3),
            noise_frac=0.0,
            seed=seed,
        )
    )


def _player_factory(agent, n_playout, eval_mode="mcts"):
    agent_type = agent.get("type")
    if agent_type == "baseline" and agent.get("id") == "random":
        return lambda _game_idx=0: RandomPlayer()
    if agent_type == "baseline" and agent.get("id") == "heuristic":
        return lambda _game_idx=0: HeuristicPlayer()
    if agent_type != "checkpoint":
        raise ValueError(f"Unsupported agent type: {agent_type}")

    policy = _checkpoint_policy(agent)
    mode = _agent_eval_mode(agent, eval_mode)
    if mode == "tactical_beam":
        return lambda game_idx=0: _beam_player(policy, agent, seed=game_idx + 10_000)
    if mode != "mcts":
        raise ValueError(f"Unsupported eval mode for checkpoint: {mode}")
    tactical_prior = _agent_mcts_tactical_prior(agent)
    tactical_leaf = _agent_mcts_tactical_leaf(agent)
    config = _agent_config(agent)
    mcts_heuristic_prior_weight = config.get("mcts_heuristic_prior_weight")

    def policy_fn(board):
        return policy.policy_value_fn(
            board,
            heuristic_prior_weight=mcts_heuristic_prior_weight,
        )

    def policy_batch_fn(boards):
        return policy.policy_value_batch_fn(
            boards,
            heuristic_prior_weight=mcts_heuristic_prior_weight,
        )

    return lambda _game_idx=0: MCTSPlayer(
            policy_fn,
            c_puct=5,
            n_playout=n_playout,
            is_selfplay=0,
            use_parallel=False,
            policy_value_batch_function=policy_batch_fn,
            mcts_batch_size=config.get("mcts_batch_size", 1),
            mcts_min_batches_per_search=config.get("mcts_min_batches_per_search", 1),
            tactical_threshold=_agent_mcts_tactical_threshold(agent),
            tactical_prior_weight=tactical_prior["weight"],
            tactical_prior_temperature=tactical_prior["temperature"],
            tactical_prior_two_ply_bonus=tactical_prior["two_ply_bonus"],
            tactical_prior_two_ply_max_candidates=tactical_prior["two_ply_max_candidates"],
            tactical_prior_two_ply_max_replies=tactical_prior["two_ply_max_replies"],
            tactical_prior_two_ply_max_followups=tactical_prior["two_ply_max_followups"],
            tactical_leaf_eval=tactical_leaf["eval"],
            tactical_leaf_win_value=tactical_leaf["win_value"],
            tactical_leaf_loss_value=tactical_leaf["loss_value"],
            tactical_leaf_forcing_value=tactical_leaf["forcing_value"],
            tactical_leaf_two_ply_value=tactical_leaf["two_ply_value"],
            tactical_leaf_two_ply=tactical_leaf["two_ply"],
            tactical_leaf_max_candidates=tactical_leaf["max_candidates"],
            tactical_leaf_max_replies=tactical_leaf["max_replies"],
            tactical_leaf_max_followups=tactical_leaf["max_followups"],
        )


def _eval_parallel_workers(candidate, games):
    config = _agent_config(candidate)
    if games <= 1:
        return 1
    return _resolve_parallel_worker_count(
        config,
        key="eval_parallel_games",
        prefix="eval",
        limit=games,
    )


def _match_game_result(
    candidate_factory,
    opponent_factory,
    board_config,
    game_idx,
):
    board = Board(
        width=board_config["board_width"],
        height=board_config["board_height"],
        n_in_row=board_config["n_in_row"],
    )
    game = Game(board)
    candidate_player = candidate_factory(game_idx)
    opponent_player = opponent_factory(game_idx)
    try:
        winner, moves = game.start_play(
            candidate_player,
            opponent_player,
            start_player=game_idx % 2,
            is_shown=0,
        )
        if winner == candidate_player.player:
            outcome = "win"
        elif winner == -1:
            outcome = "draw"
        else:
            outcome = "loss"
        return {
            "game": game_idx + 1,
            "outcome": outcome,
            "moves": len(moves),
        }
    except Exception as exc:
        return {
            "game": game_idx + 1,
            "outcome": "failure",
            "error": str(exc),
            "traceback": traceback.format_exc(limit=4),
        }


def _match_chunk(args):
    (
        candidate,
        opponent,
        game_indices,
        candidate_n_playout,
        opponent_n_playout,
        candidate_eval_mode,
        opponent_eval_mode,
    ) = args
    board_config = _board_config(candidate)
    candidate_factory = _player_factory(candidate, candidate_n_playout, candidate_eval_mode)
    opponent_factory = _player_factory(opponent, opponent_n_playout, opponent_eval_mode)
    return [
        _match_game_result(
            candidate_factory,
            opponent_factory,
            board_config,
            game_idx,
        )
        for game_idx in game_indices
    ]


def _match_chunks(games, workers):
    return [
        list(range(worker_idx, games, workers))
        for worker_idx in range(workers)
        if worker_idx < games
    ]


def _match_record(
    candidate,
    opponent,
    games,
    candidate_n_playout,
    opponent_n_playout,
    candidate_eval_mode="mcts",
    opponent_eval_mode="mcts",
):
    match_start = time.perf_counter()
    workers = _eval_parallel_workers(candidate, games)
    wins = draws = losses = failures = 0
    failure_messages = []
    move_counts = []
    outcome_move_counts = {
        "win": [],
        "draw": [],
        "loss": [],
    }

    chunks = _match_chunks(games, workers)
    if not chunks:
        result_groups = []
    elif workers == 1:
        result_groups = [
            _match_chunk(
                (
                    candidate,
                    opponent,
                    chunks[0],
                    candidate_n_playout,
                    opponent_n_playout,
                    candidate_eval_mode,
                    opponent_eval_mode,
                )
            )
        ]
    else:
        worker_args = [
            (
                candidate,
                opponent,
                chunk,
                candidate_n_playout,
                opponent_n_playout,
                candidate_eval_mode,
                opponent_eval_mode,
            )
            for chunk in chunks
        ]
        executor_kwargs = {"max_workers": workers}
        start_methods = mp.get_all_start_methods()
        for method in ("forkserver", "spawn", "fork"):
            if method in start_methods:
                executor_kwargs["mp_context"] = mp.get_context(method)
                break
        with concurrent.futures.ProcessPoolExecutor(**executor_kwargs) as executor:
            result_groups = list(executor.map(_match_chunk, worker_args))

    for results in result_groups:
        for result in results:
            outcome = result.get("outcome")
            if outcome == "win":
                wins += 1
                outcome_move_counts["win"].append(result.get("moves", 0))
            elif outcome == "draw":
                draws += 1
                outcome_move_counts["draw"].append(result.get("moves", 0))
            elif outcome == "loss":
                losses += 1
                outcome_move_counts["loss"].append(result.get("moves", 0))
            else:
                failures += 1
                failure_messages.append({
                    "game": result.get("game"),
                    "error": result.get("error"),
                    "traceback": result.get("traceback"),
                })
            if outcome in outcome_move_counts:
                move_counts.append(result.get("moves", 0))

    def average(values):
        values = [int(value or 0) for value in values]
        if not values:
            return None
        return round(sum(values) / len(values), 3)

    return {
        "opponent_id": opponent["id"],
        "opponent_name": opponent.get("name", opponent["id"]),
        "opponent_elo": opponent.get("elo", 1000),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "games": games,
        "score": score_from_record(wins, draws, losses),
        "failures": failures,
        "failure_messages": failure_messages[:3],
        "duration_s": round(time.perf_counter() - match_start, 3),
        "parallel_workers": workers,
        "avg_moves": average(move_counts),
        "win_avg_moves": average(outcome_move_counts["win"]),
        "draw_avg_moves": average(outcome_move_counts["draw"]),
        "loss_avg_moves": average(outcome_move_counts["loss"]),
    }


def _baseline_agent(agent_id, candidate):
    return {
        "id": agent_id,
        "type": "baseline",
        "name": "Random legal move" if agent_id == "random" else "One-ply heuristic",
        "elo": 800 if agent_id == "random" else 1000,
        "board_width": candidate["board_width"],
        "board_height": candidate["board_height"],
        "n_in_row": candidate["n_in_row"],
    }


def _gate_result(candidate, evaluation, gate):
    heuristic = evaluation["opponents"].get("heuristic")
    previous_best = evaluation["opponents"].get("previous_best")
    failures = sum(item.get("failures", 0) for item in evaluation["opponents"].values())

    checks = {
        "board": (
            candidate.get("board_width") == gate["board_width"]
            and candidate.get("board_height") == gate["board_height"]
            and candidate.get("n_in_row") == gate["n_in_row"]
        ),
        "heuristic_score": (
            heuristic is not None
            and heuristic["games"] >= gate["heuristic_min_games"]
            and heuristic["score"] >= gate["heuristic_min_score"]
        ),
        "previous_best_score": (
            previous_best is not None
            and previous_best["games"] >= gate["previous_best_min_games"]
            and previous_best["score"] >= gate["previous_best_min_score"]
        ),
        "elo": evaluation["elo"] >= gate["elo_target"],
        "runtime": failures <= gate["max_runtime_failures"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "gate": dict(gate),
        "summary": {
            "heuristic_score": None if heuristic is None else heuristic["score"],
            "heuristic_games": 0 if heuristic is None else heuristic["games"],
            "previous_best_score": None if previous_best is None else previous_best["score"],
            "previous_best_games": 0 if previous_best is None else previous_best["games"],
            "elo": evaluation["elo"],
            "failures": failures,
        },
    }


def evaluate_checkpoint(
    agent_id,
    registry_path="checkpoints/registry.json",
    games=40,
    previous_best_games=None,
    n_playout=None,
    opponent_n_playout=None,
    eval_mode="mcts",
    opponent_eval_mode="mcts",
    promote=True,
    gate=None,
):
    eval_start = time.perf_counter()
    candidate = find_agent(agent_id, registry_path)
    if candidate is None:
        raise ValueError(f"Unknown checkpoint '{agent_id}'")
    if candidate.get("type") != "checkpoint":
        raise ValueError("Only checkpoint agents can be evaluated for promotion")

    gate = copy.deepcopy(gate or COMPLETION_GATE)
    previous_best_games = games if previous_best_games is None else previous_best_games
    n_playout = candidate.get("n_playout", 2) if n_playout is None else n_playout
    checkpoint_opponent_override = n_playout if opponent_n_playout is None else opponent_n_playout
    baseline_opponent_n_playout = 0
    checkpoint_dir = Path(registry_path).parent
    opponents = {}

    def run_opponent(
        opponent_key,
        opponent,
        opponent_games,
        candidate_n_playout,
        current_opponent_n_playout,
        candidate_eval_mode="mcts",
        current_opponent_eval_mode="mcts",
    ):
        parallel_workers = _eval_parallel_workers(candidate, opponent_games)
        common_fields = {
            "preset": candidate.get("preset"),
            "checkpoint_id": candidate["id"],
            "games_trained": candidate.get("games_trained"),
            "opponent_key": opponent_key,
            "opponent_id": opponent["id"],
            "opponent_name": opponent.get("name", opponent["id"]),
            "opponent_elo": opponent.get("elo", 1000),
            "games": opponent_games,
            "n_playout": candidate_n_playout,
            "opponent_n_playout": current_opponent_n_playout,
            "eval_mode": candidate_eval_mode,
            "opponent_eval_mode": current_opponent_eval_mode,
            "parallel_workers": parallel_workers,
        }
        append_training_event(
            {
                "event": "evaluation_opponent_start",
                **common_fields,
                "evaluation_elapsed_s": round(time.perf_counter() - eval_start, 3),
            },
            checkpoint_dir=checkpoint_dir,
        )
        record = _match_record(
            candidate,
            opponent,
            opponent_games,
            candidate_n_playout,
            current_opponent_n_playout,
            candidate_eval_mode=candidate_eval_mode,
            opponent_eval_mode=current_opponent_eval_mode,
        )
        append_training_event(
            {
                "event": "evaluation_opponent",
                **common_fields,
                "wins": record["wins"],
                "draws": record["draws"],
                "losses": record["losses"],
                "score": record["score"],
                "failures": record["failures"],
                "failure_messages": record.get("failure_messages", []),
                "duration_s": record["duration_s"],
                "avg_moves": record.get("avg_moves"),
                "win_avg_moves": record.get("win_avg_moves"),
                "draw_avg_moves": record.get("draw_avg_moves"),
                "loss_avg_moves": record.get("loss_avg_moves"),
                "evaluation_elapsed_s": round(time.perf_counter() - eval_start, 3),
                "opponents_completed": len(opponents) + 1,
            },
            checkpoint_dir=checkpoint_dir,
        )
        return record

    opponents["random"] = run_opponent(
        "random",
        _baseline_agent("random", candidate),
        games,
        n_playout,
        baseline_opponent_n_playout,
        candidate_eval_mode=eval_mode,
        current_opponent_eval_mode="random",
    )
    opponents["heuristic"] = run_opponent(
        "heuristic",
        _baseline_agent("heuristic", candidate),
        games,
        n_playout,
        baseline_opponent_n_playout,
        candidate_eval_mode=eval_mode,
        current_opponent_eval_mode="heuristic",
    )

    champion = champion_checkpoint(candidate, registry_path, exclude_id=candidate["id"])
    champion_elo = None if champion is None else champion.get("elo", 0)
    checkpoint_opponent_n_playout = (
        opponent_n_playout
        if opponent_n_playout is not None
        else (None if champion is None else champion.get("n_playout", n_playout))
    )
    if champion is not None:
        opponents["previous_best"] = run_opponent(
            "previous_best",
            champion,
            previous_best_games,
            n_playout,
            checkpoint_opponent_n_playout,
            candidate_eval_mode=eval_mode,
            current_opponent_eval_mode=opponent_eval_mode,
        )
        opponents["previous_best"]["opponent_id"] = champion["id"]
        opponents["previous_best"]["opponent_name"] = champion.get("name", champion["id"])

    elo_inputs = [
        (record["opponent_elo"], record["score"], record["games"])
        for record in opponents.values()
        if record["games"] > 0 and record.get("failures", 0) == 0
    ]
    estimated_elo = elo_after_games(1000, elo_inputs)

    evaluation = {
        "opponents": opponents,
        "elo": estimated_elo,
        "games": games,
        "previous_best_games": previous_best_games,
        "n_playout": n_playout,
        "eval_mode": eval_mode,
        "opponent_eval_mode": opponent_eval_mode,
        "opponent_n_playout": {
            "baseline": baseline_opponent_n_playout,
            "previous_best": checkpoint_opponent_n_playout,
            "checkpoint_override": checkpoint_opponent_override,
        },
    }
    gate_result = _gate_result(candidate, evaluation, gate)

    promotion_gate = copy.deepcopy(PROMOTION_GATE)
    heuristic_record = opponents.get("heuristic")
    previous_best_record = opponents.get("previous_best")
    failures = sum(item.get("failures", 0) for item in opponents.values())
    min_elo_delta = promotion_gate.get("min_elo_delta")
    elo_delta = None if champion_elo is None else estimated_elo - champion_elo
    elo_improved = champion_elo is None or estimated_elo > champion_elo
    elo_floor_passed = (
        champion_elo is None
        or min_elo_delta is None
        or elo_delta >= min_elo_delta
    )
    head_to_head_improved = (
        champion is None
        or (
            previous_best_record is not None
            and previous_best_record["score"] > 0.5
        )
    )
    heuristic_games_ok = (
        heuristic_record is not None
        and heuristic_record["games"] >= promotion_gate["heuristic_min_games"]
    )
    heuristic_passed = (
        heuristic_games_ok
        and heuristic_record["score"] >= promotion_gate["heuristic_min_score"]
    )
    previous_best_games_ok = (
        previous_best_record is None
        or previous_best_record["games"] >= promotion_gate["previous_best_min_games"]
    )
    previous_best_passed = (
        previous_best_record is None
        or (
            previous_best_games_ok
            and previous_best_record["score"] >= promotion_gate["previous_best_min_score"]
        )
    )
    improvement_passed = elo_floor_passed and head_to_head_improved
    promotion_passed = (
        failures <= promotion_gate["max_runtime_failures"]
        and heuristic_passed
        and previous_best_games_ok
        and previous_best_passed
        and improvement_passed
    )
    if promotion_passed:
        reason = "improved the local ladder and passed promotion checks"
    elif failures > promotion_gate["max_runtime_failures"]:
        reason = "runtime failures during evaluation"
    elif not heuristic_games_ok:
        reason = "not enough heuristic games for promotion"
    elif not heuristic_passed:
        reason = "did not pass heuristic promotion floor"
    elif not previous_best_games_ok:
        reason = "not enough previous-best games for promotion"
    elif not previous_best_passed:
        reason = "did not beat previous best"
    elif not elo_floor_passed:
        reason = "fixed-baseline Elo regressed versus champion"
    elif not improvement_passed:
        reason = "did not improve local ladder enough to replace champion"
    else:
        reason = "promotion checks failed"

    promotion = {
        "promoted": bool(promotion_passed),
        "reason": reason,
        "gate_passed": gate_result["passed"],
        "champion_elo": champion_elo,
        "elo_delta": elo_delta,
        "min_elo_delta": min_elo_delta,
        "elo_improved": elo_improved,
        "elo_floor_passed": elo_floor_passed,
        "head_to_head_improved": head_to_head_improved,
        "promotion_gate": promotion_gate,
        "promotion_checks": {
            "runtime": failures <= promotion_gate["max_runtime_failures"],
            "heuristic_games": heuristic_games_ok,
            "heuristic_score": heuristic_passed,
            "previous_best_games": previous_best_games_ok,
            "previous_best_score": previous_best_passed,
            "elo_floor": elo_floor_passed,
            "improvement": improvement_passed,
            "elo_improved": elo_improved,
            "head_to_head_improved": head_to_head_improved,
        },
        "completion_gate": gate_result,
        "average_human_gate": gate_result,
    }

    if promote:
        update_checkpoint(
            candidate["id"],
            registry_path=registry_path,
            updates={"elo": estimated_elo},
            metrics_updates={
                "evaluation": evaluation,
                "promotion": promotion,
                "eval": opponents,
                "elo": estimated_elo,
            },
        )
    append_training_event(
        {
            "event": "evaluation",
            "preset": candidate.get("preset"),
            "checkpoint_id": candidate["id"],
            "games_trained": candidate.get("games_trained"),
            "elo": estimated_elo,
            "games": games,
            "previous_best_games": previous_best_games,
            "duration_s": round(time.perf_counter() - eval_start, 3),
            "evaluation": evaluation,
            "promotion": promotion,
            "promote": bool(promote),
        },
        checkpoint_dir=Path(registry_path).parent,
    )

    return {
        "candidate_id": candidate["id"],
        "candidate_name": candidate.get("name", candidate["id"]),
        "evaluation": evaluation,
        "promotion": promotion,
    }
