import os

for _thread_env in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")

import argparse
from config import IMPROVEMENT_PRESETS, preset_names
from evaluator import evaluate_checkpoint
from improve import run_improvement_loop
from train import TrainPipeline, run_baseline_training
from checkpoint_registry import find_agent
from game import Board, Game
from mcts import MCTSPlayer
from model import PolicyValueNet

def run_training(
    preset="shopping_baseline",
    debug=False,
    resume_best=False,
    init_agent_id=None,
    max_runtime_minutes=None,
    max_runtime_dispatch_margin_minutes=None,
):
    if debug:
        preset = "debug"
    if preset == "full":
        print("Starting full legacy training...")
        pipeline = TrainPipeline(debug=debug)
        pipeline.run()
    else:
        init_agent = find_agent(init_agent_id) if init_agent_id else None
        print(f"Starting bounded baseline training with preset '{preset}'...")
        result = run_baseline_training(
            preset=preset,
            init_agent=init_agent,
            resume_best=resume_best,
            max_runtime_minutes=max_runtime_minutes,
            max_runtime_dispatch_margin_minutes=max_runtime_dispatch_margin_minutes,
        )
        final = result["final"]
        print(f"Saved final checkpoint: {final['path']}")
        print(f"Estimated Elo: {final['elo']}")

def play_game(model_file=None):
    print("Starting game...")
    board = Board(width=15, height=15, n_in_row=5)
    game = Game(board)
    
    if model_file:
        policy_value_net = PolicyValueNet(15, 15, model_file=model_file)
        mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=5, n_playout=400)
    else:
        print("No model provided, using random MCTS.")
        # Just a dummy function for random play if needed, or handle otherwise
        # For now, let's assume we always want a model for the AI
        policy_value_net = PolicyValueNet(15, 15)
        mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=5, n_playout=400)

    # Human player logic would go here, or simple AI vs AI
    # For simplicity, let's do AI vs AI self-play visualization
    game.start_self_play(mcts_player, is_shown=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gomoku AlphaZero')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'play', 'web', 'evaluate', 'improve'], help='Mode: train, play, web, evaluate, or improve')
    parser.add_argument('--model', type=str, default=None, help='Path to model file for play mode')
    parser.add_argument('--agent', type=str, default=None, help='Checkpoint id for evaluate mode')
    parser.add_argument('--init-agent', type=str, default=None, help='Checkpoint id to initialize bounded train mode')
    parser.add_argument('--resume-best', action='store_true', help='Initialize bounded train mode from the best compatible checkpoint')
    parser.add_argument('--fresh', action='store_true', help='Do not resume from best checkpoint in improve mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (low resource, logging)')
    parser.add_argument('--preset', type=str, default='shopping_baseline', choices=preset_names(), help='Training preset')
    parser.add_argument('--presets', type=str, default=None, help='Comma-separated presets for improve mode')
    parser.add_argument('--eval-games', type=int, default=16, help='Evaluation games per opponent for evaluate/improve mode')
    parser.add_argument('--previous-best-games', type=int, default=None, help='Games against previous best checkpoint')
    parser.add_argument('--eval-n-playout', type=int, default=None, help='Candidate MCTS playouts for evaluate/improve diagnostics')
    parser.add_argument('--opponent-eval-n-playout', type=int, default=None, help='Opponent MCTS playouts for evaluate/improve diagnostics')
    parser.add_argument('--eval-mode', type=str, default='mcts', choices=['mcts', 'native', 'tactical_beam'], help='Candidate player mode for evaluate diagnostics')
    parser.add_argument('--opponent-eval-mode', type=str, default='mcts', choices=['mcts', 'native', 'tactical_beam'], help='Checkpoint opponent player mode for evaluate diagnostics')
    parser.add_argument('--no-promote', action='store_true', help='Run evaluate mode without updating promotion/champion registry fields')
    parser.add_argument('--rounds', type=int, default=1, help='Improvement rounds to run over the selected presets')
    parser.add_argument('--max-runtime-minutes', type=float, default=None, help='Stop training after this wall-clock budget, then save replay/checkpoint and continue cleanup/eval')
    parser.add_argument('--max-runtime-dispatch-margin-minutes', type=float, default=None, help='Stop dispatching new self-play games this many minutes before the runtime budget is projected to expire')
    parser.add_argument('--keep-going-after-gate', action='store_true', help='Continue improve mode even after the completion gate passes')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Web server host')
    parser.add_argument('--port', type=int, default=8000, help='Web server port')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        run_training(
            preset=args.preset,
            debug=args.debug,
            resume_best=args.resume_best,
            init_agent_id=args.init_agent,
            max_runtime_minutes=args.max_runtime_minutes,
            max_runtime_dispatch_margin_minutes=args.max_runtime_dispatch_margin_minutes,
        )
    elif args.mode == 'play':
        play_game(args.model)
    elif args.mode == 'web':
        from web_app import run_server
        run_server(host=args.host, port=args.port)
    elif args.mode == 'evaluate':
        if not args.agent:
            parser.error("--agent is required for evaluate mode")
        result = evaluate_checkpoint(
            args.agent,
            games=args.eval_games,
            previous_best_games=args.previous_best_games,
            n_playout=args.eval_n_playout,
            opponent_n_playout=args.opponent_eval_n_playout,
            eval_mode=args.eval_mode,
            opponent_eval_mode=args.opponent_eval_mode,
            promote=not args.no_promote,
        )
        print(f"Evaluated {result['candidate_name']}")
        print(f"Estimated Elo: {result['evaluation']['elo']}")
        print(f"Promoted: {result['promotion']['promoted']}")
        print(f"Completion gate: {result['promotion']['gate_passed']}")
    else:
        presets = IMPROVEMENT_PRESETS if args.presets is None else tuple(
            item.strip() for item in args.presets.split(',') if item.strip()
        )
        results = run_improvement_loop(
            presets=presets,
            eval_games=args.eval_games,
            previous_best_games=args.previous_best_games,
            resume_best=not args.fresh,
            rounds=args.rounds,
            stop_on_gate=not args.keep_going_after_gate,
            max_runtime_minutes=args.max_runtime_minutes,
            max_runtime_dispatch_margin_minutes=args.max_runtime_dispatch_margin_minutes,
            eval_n_playout=args.eval_n_playout,
            opponent_eval_n_playout=args.opponent_eval_n_playout,
            eval_mode=args.eval_mode,
            opponent_eval_mode=args.opponent_eval_mode,
        )
        for result in results:
            promotion = result["evaluation"]["promotion"]
            checkpoint = result["checkpoint"]
            print(
                f"round={result['round']} {result['preset']}: {checkpoint['id']} "
                f"elo={result['evaluation']['evaluation']['elo']} "
                f"promoted={promotion['promoted']} "
                f"gate={promotion['gate_passed']}"
            )
