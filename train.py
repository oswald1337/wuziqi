import logging
import copy
import json
import time
import os
import torch
import random
import pickle
import numpy as np
import multiprocessing as mp
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from config import get_training_preset
from game import Board, Game
from mcts import MCTS, MCTSPlayer
from model import PolicyValueNet
from checkpoint_registry import (
    DEFAULT_REGISTRY_PATH,
    best_compatible_model_checkpoint,
    checkpoint_architecture,
    elo_after_games,
    register_checkpoint,
    resolve_model_path,
    score_from_record,
)
from players import HeuristicPlayer, RandomPlayer
from tactical import (
    DIRECTIONS,
    OPEN_FOUR_SCORE,
    WIN_SCORE,
    best_forcing_win_move,
    best_tactical_move,
    creates_bounded_two_ply_threat,
    creates_unanswerable_threat,
    fork_threat_count,
    line_shape,
    plausible_reply_moves,
    ranked_tactical_moves,
    winning_moves,
)
from tracking import append_training_event

wandb = None


def load_wandb():
    global wandb
    if wandb is not None:
        return wandb
    try:
        import wandb as wandb_module
    except ImportError:
        return None
    wandb = wandb_module
    return wandb


def augment_play_data(play_data, board_height, board_width):
    """Augment self-play samples by board rotations and flips."""
    extend_data = []
    for state, mcts_prob, winner in play_data:
        for i in [1, 2, 3, 4]:
            equi_state = np.array([np.rot90(s, i) for s in state])
            equi_mcts_prob = np.rot90(np.flipud(
                mcts_prob.reshape(board_height, board_width)), i)
            extend_data.append((
                equi_state,
                np.flipud(equi_mcts_prob).flatten(),
                winner,
            ))
            equi_state = np.array([np.fliplr(s) for s in equi_state])
            equi_mcts_prob = np.fliplr(equi_mcts_prob)
            extend_data.append((
                equi_state,
                np.flipud(equi_mcts_prob).flatten(),
                winner,
            ))
    return extend_data


def wandb_log(payload):
    if wandb is not None and getattr(wandb, "run", None) is not None:
        wandb.log(payload)


def wandb_save(path):
    if wandb is not None and getattr(wandb, "run", None) is not None:
        wandb.save(path)

# Configure logging
def setup_logger(log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    # Remove existing handlers to avoid duplicates if called multiple times
    logger.handlers = []
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    return logger

class PerformanceMonitor:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.inference_times = []
        self.wait_times = []
        self.training_times = []
        self.batch_sizes = []
        self.start_time = time.time()
        
    def log_inference(self, duration, batch_size):
        self.inference_times.append(duration)
        self.batch_sizes.append(batch_size)
        
    def log_wait(self, duration):
        self.wait_times.append(duration)
        
    def log_training(self, duration):
        self.training_times.append(duration)
        
    def get_stats(self):
        stats = {}
        if self.inference_times:
            stats['avg_inference_ms'] = np.mean(self.inference_times) * 1000
            stats['max_inference_ms'] = np.max(self.inference_times) * 1000
        if self.wait_times:
            stats['avg_wait_ms'] = np.mean(self.wait_times) * 1000
            stats['total_wait_s'] = np.sum(self.wait_times)
        if self.training_times:
            stats['avg_train_s'] = np.mean(self.training_times)
        if self.batch_sizes:
            stats['avg_batch_size'] = np.mean(self.batch_sizes)
            
        stats['duration_s'] = time.time() - self.start_time
        return stats

def self_play_worker(worker_id, conn, config, model_file=None):
    """
    Worker process that plays games against itself.
    conn: Connection to the model server (Pipe)
    """
    # Setup worker logger to write to main train.log
    worker_logger = logging.getLogger(f"worker_{worker_id}")
    worker_logger.setLevel(logging.INFO)
    handler = logging.FileHandler("train.log")
    handler.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(levelname)s %(message)s'))
    worker_logger.addHandler(handler)
    
    worker_logger.info(f"Worker {worker_id} started")
    
    # Initialize environment
    board = Board(width=config['board_width'],
                  height=config['board_height'],
                  n_in_row=config['n_in_row'])
    game = Game(board)
    
    # Track statistics
    request_count = 0
    
    # Define the remote policy function
    def policy_value_fn(board):
        nonlocal request_count
        request_count += 1
        # Send request to server
        conn.send((board.current_state(), board.availables))
        # Wait for response
        action_probs, value = conn.recv()
        return action_probs, value

    # Initialize MCTS player with remote policy
    mcts_player = MCTSPlayer(policy_value_fn,
                             c_puct=config['c_puct'],
                             n_playout=config['n_playout'],
                             is_selfplay=1,
                             use_parallel=False)
    
    game_num = 0
    while True:
        # Play a game
        game_start = time.time()
        request_count = 0
        
        winner, play_data, moves = game.start_self_play(mcts_player, temp=config['temp'])
        
        game_duration = time.time() - game_start
        game_num += 1
        
        worker_logger.info(f"Worker {worker_id} Game {game_num}: {game_duration:.2f}s, "
                          f"{len(moves)} moves, {request_count} NN requests, winner: {winner}")
        
        # Send game data to server
        conn.send(("DATA", (winner, list(play_data), moves)))
        
        # No response expected for data

def evaluation_worker(worker_id, conn, config, best_model_file, current_model_file):
    """
    Worker for evaluating current model against best model.
    This is a bit complex because we need TWO models.
    Simplification: The server holds the 'current' model.
    We can have the server handle inference for 'current' player.
    But for 'best' player, we might need another model instance or server?
    
    Alternative: Just load the models here on CPU? Or GPU if available?
    If we have 1 GPU, we can't easily share it across processes without a server.
    
    Let's stick to the plan: "Play N games between them."
    If we want to use the GPU server for both, we need to distinguish requests.
    
    For now, to keep it simple, let's load the models on CPU in this worker. 
    Evaluation is less frequent, so CPU inference might be acceptable.
    Or, if we have enough VRAM, we can load them on GPU here too? 
    But CUDA context sharing is tricky.
    
    Let's try CPU inference for evaluation to avoid complexity.
    """
    try:
        board = Board(width=config['board_width'],
                      height=config['board_height'],
                      n_in_row=config['n_in_row'])
        game = Game(board)
        
        # Load policies
        # We need to handle the case where files might not exist yet or are being written
        time.sleep(1) # Wait a bit for file sync
        
        if not os.path.exists(best_model_file) or not os.path.exists(current_model_file):
            return 0.0
            
        policy_best = PolicyValueNet(config['board_width'], config['board_height'], model_file=best_model_file, use_gpu=False)
        policy_curr = PolicyValueNet(config['board_width'], config['board_height'], model_file=current_model_file, use_gpu=False)
        
        mcts_best = MCTSPlayer(policy_best.policy_value_fn, c_puct=config['c_puct'], n_playout=config['n_playout'])
        mcts_curr = MCTSPlayer(policy_curr.policy_value_fn, c_puct=config['c_puct'], n_playout=config['n_playout'])
        
        win_cnt = 0
        n_games = 10
        for i in range(n_games):
            # start_player=0 -> current goes first
            # start_player=1 -> best goes first
            # We want to alternate
            winner, _moves = game.start_play(mcts_curr, mcts_best, start_player=i % 2, is_shown=0)
            if winner == mcts_curr.player:
                win_cnt += 1
        
        return win_cnt / n_games
    except Exception as e:
        print(f"Eval error: {e}")
        return 0.0

class TrainPipeline:
    def __init__(self, init_model=None, debug=False):
        self.board_width = 15
        self.board_height = 15
        self.n_in_row = 5
        self.debug = debug
        
        # Logging
        self.logger = setup_logger("train.log")
        self.logger = logging.getLogger("master")  # Rename to 'master' for clarity
        self.monitor = PerformanceMonitor()
        
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0
        self.temp = 1.0
        self.n_playout = 400
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.epochs = 5
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        
        # Number of self-play workers based on CPU count
        self.num_workers = mp.cpu_count()
        if self.debug:
            self.num_workers = 2
            self.n_playout = 50
            self.batch_size = 2
            self.buffer_size = 100
            self.epochs = 1
            self.check_freq = 2
            self.game_batch_num = 10
            
        self.config = {
            "board_width": self.board_width,
            "board_height": self.board_height,
            "n_in_row": self.n_in_row,
            "n_playout": self.n_playout,
            "c_puct": self.c_puct,
            "temp": self.temp
        }

        # Initialize model
        if init_model:
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, model_file=init_model)
        else:
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height)
            
        # Save initial models
        self.policy_value_net.save_model('./current_policy.model')
        self.policy_value_net.save_model('./best_policy.model')

        # WandB is optional for the local baseline path. Keep the legacy trainer
        # usable without forcing a login in a fresh checkout.
        wandb_module = load_wandb()
        if wandb_module is not None:
            wandb_mode = os.environ.get("WANDB_MODE")
            if wandb_mode is None and not os.environ.get("WANDB_API_KEY"):
                wandb_mode = "disabled"
            wandb_module.init(project="gomoku-rl", config=self.config, mode=wandb_mode)
        else:
            self.logger.warning("wandb is not installed; metrics will be logged locally only.")
        
        self.logger.info(f"Training started with config: {self.config}")

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping"""
        return augment_play_data(play_data, self.board_height, self.board_width)

    def policy_update(self):
        """update the policy-value net"""
        start_time = time.time()
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1))
            if kl > self.kl_targ * 4:
                break
        
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        wandb_log({
            "loss": loss,
            "entropy": entropy,
            "kl": kl,
            "lr_multiplier": self.lr_multiplier,
            "learning_rate": self.learn_rate * self.lr_multiplier
        })
        
        self.monitor.log_training(time.time() - start_time)
        return loss, entropy

    def run(self):
        # Start workers
        workers = []
        pipes = []
        
        for i in range(self.num_workers):
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(target=self_play_worker, args=(i, child_conn, self.config))
            p.start()
            workers.append(p)
            pipes.append(parent_conn)
            
        self.logger.info(f"Started {self.num_workers} workers (CPU count: {mp.cpu_count()})")
        
        try:
            game_count = 0
            while game_count < self.game_batch_num:
                # Model Server Loop
                # Collect requests
                wait_start = time.time()
                ready_pipes = mp.connection.wait(pipes, timeout=0.01)
                self.monitor.log_wait(time.time() - wait_start)
                
                requests = []
                request_pipes = []
                request_availables = []
                
                for pipe in ready_pipes:
                    try:
                        msg = pipe.recv()
                        # Check if msg is data using strict type checking to avoid numpy ambiguity
                        if isinstance(msg, tuple) and isinstance(msg[0], str) and msg[0] == "DATA":
                            # Handle game data
                            _, (winner, play_data, moves) = msg
                            play_data = self.get_equi_data(play_data)
                            self.data_buffer.extend(play_data)
                            game_count += 1
                            self.logger.info(f"Game {game_count} collected. Buffer size: {len(self.data_buffer)}")
                            
                            # Log sample game
                            if game_count % 10 == 0:
                                from html_logger import HtmlLogger
                                logger = HtmlLogger()
                                logger.save_game(moves, winner, filename=f"game_{game_count}.html")
                                if wandb is not None and getattr(wandb, "run", None) is not None:
                                    wandb.log({"game_replay": wandb.Html(open(f"logs/game_{game_count}.html"))})

                            # Check for training
                            if len(self.data_buffer) > self.batch_size:
                                self.policy_update()
                                
                            # Check for evaluation
                            if game_count % self.check_freq == 0:
                                self.policy_value_net.save_model('./current_policy.model')
                                self.logger.info("Evaluating...")
                                win_ratio = evaluation_worker(0, None, self.config, './best_policy.model', './current_policy.model')
                                self.logger.info(f"Win ratio: {win_ratio}")
                                wandb_log({"win_ratio": win_ratio})
                                
                                if win_ratio > 0.55: # Slight bias towards challenger
                                    self.logger.info("New best policy!")
                                    self.best_win_ratio = win_ratio
                                    self.policy_value_net.save_model('./best_policy.model')
                                    
                                    # Log model artifact to WandB
                                    if wandb is not None and getattr(wandb, "run", None) is not None:
                                        artifact = wandb.Artifact('gomoku-policy', type='model')
                                        artifact.add_file('./best_policy.model')
                                        wandb.log_artifact(artifact)
                                
                        else:
                            # Prediction request: (state, availables)
                            state, availables = msg
                            requests.append(state)
                            request_availables.append(availables)
                            request_pipes.append(pipe)
                            self.logger.info(f"Requests growing: current size {len(requests)}")
                    except EOFError:
                        pass
                
                # Batch inference
                if requests:
                    inference_start = time.time()
                    state_batch = np.array(requests)
                    self.logger.info(f"Performing inference on device: {next(self.policy_value_net.policy_value_net.parameters()).device}")
                    act_probs, values = self.policy_value_net.policy_value(state_batch)
                    self.monitor.log_inference(time.time() - inference_start, len(requests))
                    
                    for i, pipe in enumerate(request_pipes):
                        # Filter legal moves
                        legal_moves = request_availables[i]
                        probs = act_probs[i]
                        # We need to return a list of (action, prob) tuples
                        legal_probs = list(zip(legal_moves, probs[legal_moves]))
                        
                        pipe.send((legal_probs, values[i]))
                
                # Log stats periodically
                if game_count > 0 and game_count % 100 == 0:
                    stats = self.monitor.get_stats()
                    self.logger.info(f"Stats at game {game_count}: {stats}")
                    wandb_log(stats)
                    self.monitor.reset()
                
        except KeyboardInterrupt:
            self.logger.info("Stopping...")
        finally:
            # Upload log file to WandB
            if os.path.exists("train.log"):
                wandb_save("train.log")
                self.logger.info("Uploaded train.log to WandB")
            
            for p in workers:
                p.terminate()
                p.join()


def _train_step_from_buffer(
    policy_value_net,
    data_buffer,
    config,
    policy_loss_weight=None,
    value_loss_weight=None,
    priority_buffer=None,
    priority_fraction=None,
):
    batch_size = min(config["batch_size"], len(data_buffer))
    mini_batch, priority_samples = _sample_training_batch(
        data_buffer,
        batch_size,
        priority_buffer=priority_buffer,
        priority_fraction=(
            config.get("conversion_replay_fraction", 0.0)
            if priority_fraction is None
            else priority_fraction
        ),
    )
    state_batch = [data[0] for data in mini_batch]
    mcts_probs_batch = [data[1] for data in mini_batch]
    winner_batch = [data[2] for data in mini_batch]

    loss = 0.0
    entropy = 0.0
    for _ in range(config["epochs"]):
        loss, entropy = policy_value_net.train_step(
            state_batch,
            mcts_probs_batch,
            winner_batch,
            config["learn_rate"],
            policy_loss_weight=(
                config.get("policy_loss_weight", 1.0)
                if policy_loss_weight is None
                else policy_loss_weight
            ),
            value_loss_weight=(
                config.get("value_loss_weight", 1.0)
                if value_loss_weight is None
                else value_loss_weight
            ),
        )
    return {
        "loss": loss,
        "entropy": entropy,
        "batch_size": batch_size,
        "priority_samples": priority_samples,
        **getattr(policy_value_net, "last_train_components", {}),
    }


def _sample_training_batch(
    data_buffer,
    batch_size,
    priority_buffer=None,
    priority_fraction=0.0,
):
    if batch_size <= 0:
        return [], 0

    priority_batch = []
    priority_fraction = max(0.0, min(1.0, float(priority_fraction or 0.0)))
    if priority_buffer and priority_fraction > 0.0:
        priority_count = min(
            len(priority_buffer),
            batch_size,
            int(round(batch_size * priority_fraction)),
        )
        if priority_count > 0:
            priority_batch = random.sample(list(priority_buffer), priority_count)

    base_count = batch_size - len(priority_batch)
    base_batch = random.sample(data_buffer, min(base_count, len(data_buffer)))
    if len(base_batch) < base_count and priority_buffer:
        extra_count = min(
            base_count - len(base_batch),
            len(priority_buffer),
        )
        priority_batch.extend(random.sample(list(priority_buffer), extra_count))

    return priority_batch + base_batch, len(priority_batch)


def _conversion_replay_train_config(config):
    conversion_config = dict(config)
    conversion_config["batch_size"] = int(
        config.get("conversion_replay_batch_size")
        or config.get("batch_size")
        or 1
    )
    conversion_config["epochs"] = int(config.get("conversion_replay_epochs", 1) or 1)
    return conversion_config


def _conversion_teacher_train_config(config):
    teacher_config = dict(config)
    teacher_config["batch_size"] = int(
        config.get("conversion_teacher_batch_size")
        or config.get("batch_size")
        or 1
    )
    teacher_config["epochs"] = int(config.get("conversion_teacher_epochs", 1) or 1)
    return teacher_config


def _threat_space_proof_train_config(config):
    proof_config = dict(config)
    proof_config["batch_size"] = int(
        config.get("threat_space_proof_batch_size")
        or config.get("batch_size")
        or 1
    )
    proof_config["epochs"] = int(config.get("threat_space_proof_epochs", 1) or 1)
    return proof_config


def _mcts_distill_train_config(config):
    distill_config = dict(config)
    distill_config["batch_size"] = int(
        config.get("mcts_distill_batch_size")
        or config.get("batch_size")
        or 1
    )
    distill_config["epochs"] = int(config.get("mcts_distill_epochs", 1) or 1)
    return distill_config


def _conversion_teacher_data(config, moves, value=0.95):
    board = _new_board(config)
    board.init_board()
    teacher_data = []
    value = float(value)
    depth = config.get("conversion_teacher_depth", "one_ply")
    reply_limit = int(config.get("conversion_teacher_reply_limit", 8) or 0)
    followup_limit = int(config.get("conversion_teacher_followup_limit", 16) or 0)
    for move in moves:
        if move not in board.availables:
            break
        current = board.get_current_player()
        is_one_ply = creates_unanswerable_threat(board, move, current)
        is_teacher_move = is_one_ply
        if depth not in {"one_ply", "one_or_two_ply", "two_ply"}:
            raise ValueError(f"Unknown conversion_teacher_depth: {depth}")
        if depth == "two_ply":
            is_teacher_move = creates_bounded_two_ply_threat(
                board,
                move,
                current,
                max_replies=reply_limit,
                max_followups=followup_limit,
                include_one_ply=False,
            )
        elif depth == "one_or_two_ply" and not is_one_ply:
            is_teacher_move = creates_bounded_two_ply_threat(
                board,
                move,
                current,
                max_replies=reply_limit,
                max_followups=followup_limit,
                include_one_ply=False,
            )
        if is_teacher_move:
            policy_target = np.zeros(board.width * board.height)
            policy_target[move] = 1.0
            teacher_data.append((board.current_state(), policy_target, value))
        board.do_move(move)
        end, _winner = board.game_end()
        if end:
            break
    return teacher_data


TRAIN_METRIC_KEYS = (
    "loss",
    "entropy",
    "batch_size",
    "policy_loss",
    "value_loss",
    "policy_loss_weight",
    "value_loss_weight",
    "priority_samples",
)


def _train_metric_fields(metrics):
    fields = {}
    for key in TRAIN_METRIC_KEYS:
        if key not in metrics:
            continue
        value = metrics[key]
        fields[key] = int(value) if key in {"batch_size", "priority_samples"} else float(value)
    return fields


def _new_board(config):
    return Board(
        width=config["board_width"],
        height=config["board_height"],
        n_in_row=config["n_in_row"],
    )


def _evaluate_policy(policy_value_net, config):
    results = {}
    opponents = [
        ("random", 800, lambda: RandomPlayer()),
        ("heuristic", 1000, lambda: HeuristicPlayer()),
    ]

    for opponent_id, opponent_elo, opponent_factory in opponents:
        wins = 0
        draws = 0
        losses = 0
        for game_idx in range(config["eval_games"]):
            board = _new_board(config)
            game = Game(board)
            model_player = MCTSPlayer(
                policy_value_net.policy_value_fn,
                c_puct=config["c_puct"],
                n_playout=config["eval_n_playout"],
                is_selfplay=0,
                use_parallel=False,
                tactical_threshold=config.get("mcts_tactical_threshold"),
                two_ply_threats=config.get("mcts_two_ply_threats", False),
                two_ply_max_candidates=config.get("mcts_two_ply_max_candidates", 16),
                two_ply_max_replies=config.get("mcts_two_ply_max_replies", 6),
                two_ply_max_followups=config.get("mcts_two_ply_max_followups", 12),
                tactical_prior_weight=config.get("mcts_tactical_prior_weight", 0.0),
                tactical_prior_temperature=config.get("mcts_tactical_prior_temperature", 1.0),
                tactical_prior_two_ply_bonus=config.get("mcts_tactical_prior_two_ply_bonus", 0.0),
                tactical_prior_two_ply_max_candidates=config.get("mcts_tactical_prior_two_ply_max_candidates", 16),
                tactical_prior_two_ply_max_replies=config.get("mcts_tactical_prior_two_ply_max_replies", 6),
                tactical_prior_two_ply_max_followups=config.get("mcts_tactical_prior_two_ply_max_followups", 12),
                tactical_leaf_eval=config.get("mcts_tactical_leaf_eval", False),
                tactical_leaf_win_value=config.get("mcts_tactical_leaf_win_value", 1.0),
                tactical_leaf_loss_value=config.get("mcts_tactical_leaf_loss_value", 0.95),
                tactical_leaf_forcing_value=config.get("mcts_tactical_leaf_forcing_value", 0.85),
                tactical_leaf_two_ply_value=config.get("mcts_tactical_leaf_two_ply_value", 0.70),
                tactical_leaf_two_ply=config.get("mcts_tactical_leaf_two_ply", False),
                tactical_leaf_max_candidates=config.get("mcts_tactical_leaf_max_candidates", 16),
                tactical_leaf_max_replies=config.get("mcts_tactical_leaf_max_replies", 6),
                tactical_leaf_max_followups=config.get("mcts_tactical_leaf_max_followups", 12),
            )
            opponent = opponent_factory()
            winner, _moves = game.start_play(
                model_player,
                opponent,
                start_player=game_idx % 2,
                is_shown=0,
            )
            if winner == model_player.player:
                wins += 1
            elif winner == -1:
                draws += 1
            else:
                losses += 1

        score = score_from_record(wins, draws, losses)
        results[opponent_id] = {
            "opponent_elo": opponent_elo,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "games": config["eval_games"],
            "score": score,
        }

    elo = elo_after_games(
        1000,
        [
            (item["opponent_elo"], item["score"], item["games"])
            for item in results.values()
        ],
    )
    return results, elo


class PolicyTacticalSelfPlayPlayer:
    """Cheap self-play player for early training before MCTS is worth its cost."""

    def __init__(
        self,
        policy_value_net,
        tactical_guard=True,
        dirichlet_alpha=0.3,
        noise_frac=0.15,
        seed=None,
    ):
        self.policy_value_net = policy_value_net
        self.tactical_guard = tactical_guard
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_frac = noise_frac
        self._rng = np.random.default_rng(seed)
        self.player = None
        self.forced_tactical_moves = 0
        self.threat_solver_moves = 0
        self.policy_moves = 0

    def set_player_ind(self, player):
        self.player = player

    def reset_player(self):
        pass

    def get_action(self, board, temp=1.0, return_prob=0):
        move_probs = np.zeros(board.width * board.height)
        if not board.availables:
            return (None, move_probs) if return_prob else None

        forced = (
            best_tactical_move(board, return_reason=True)
            if self.tactical_guard
            else {"move": None, "reason": None}
        )
        forced_move = forced["move"]
        if forced_move is not None:
            move_probs[forced_move] = 1.0
            self.forced_tactical_moves += 1
            if forced["reason"] == "forcing_win":
                self.threat_solver_moves += 1
            return (forced_move, move_probs) if return_prob else forced_move

        action_probs, _value = self.policy_value_net.policy_value_fn(board)
        pairs = list(action_probs)
        acts = np.array([move for move, _prob in pairs], dtype=np.int64)
        probs = np.array([prob for _move, prob in pairs], dtype=np.float64)
        prob_sum = float(np.sum(probs))
        if not np.isfinite(prob_sum) or prob_sum <= 0:
            probs = np.full(len(acts), 1.0 / len(acts))
        else:
            probs = probs / prob_sum

        if temp <= 1e-3:
            move = int(acts[int(np.argmax(probs))])
            move_probs[acts] = probs
            self.policy_moves += 1
            return (move, move_probs) if return_prob else move

        if abs(temp - 1.0) > 1e-6:
            probs = np.power(np.maximum(probs, 1e-12), 1.0 / temp)
            probs = probs / np.sum(probs)

        if self.noise_frac > 0 and len(acts) > 1:
            alpha = max(float(self.dirichlet_alpha), 1e-6)
            noise = self._rng.dirichlet(np.full(len(acts), alpha))
            probs = (1.0 - self.noise_frac) * probs + self.noise_frac * noise
            probs = probs / np.sum(probs)

        move = int(self._rng.choice(acts, p=probs))
        move_probs[acts] = probs
        self.policy_moves += 1
        return (move, move_probs) if return_prob else move


class TacticalBeamSelfPlayPlayer:
    """One-ply tactical/value beam for early staged search without MCTS cost."""

    def __init__(
        self,
        policy_value_net,
        tactical_guard=True,
        beam_width=8,
        policy_top_k=8,
        value_weight=1.2,
        policy_weight=0.45,
        tactical_weight=1.0,
        reply_penalty=0.35,
        fork_weight=0.0,
        fork_threshold=OPEN_FOUR_SCORE,
        dirichlet_alpha=0.3,
        noise_frac=0.10,
        seed=None,
    ):
        self.policy_value_net = policy_value_net
        self.tactical_guard = tactical_guard
        self.beam_width = max(1, int(beam_width))
        self.policy_top_k = max(1, int(policy_top_k))
        self.value_weight = float(value_weight)
        self.policy_weight = float(policy_weight)
        self.tactical_weight = float(tactical_weight)
        self.reply_penalty = float(reply_penalty)
        self.fork_weight = float(fork_weight)
        self.fork_threshold = float(fork_threshold)
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_frac = noise_frac
        self._rng = np.random.default_rng(seed)
        self.player = None
        self.forced_tactical_moves = 0
        self.threat_solver_moves = 0
        self.beam_moves = 0
        self.fork_moves = 0
        self.candidate_evaluations = 0

    def set_player_ind(self, player):
        self.player = player

    def reset_player(self):
        pass

    def _candidate_value(self, board, move, current_player):
        child = copy.deepcopy(board)
        child.do_move(move)
        end, winner = child.game_end()
        if end:
            if winner == -1:
                return 0.0, False
            return (1.0 if winner == current_player else -1.0), False

        _child_probs, child_value = self.policy_value_net.policy_value_fn(child)
        reply = best_tactical_move(child) is not None
        return -float(child_value), reply

    def get_action(self, board, temp=1.0, return_prob=0):
        move_probs = np.zeros(board.width * board.height)
        if not board.availables:
            return (None, move_probs) if return_prob else None

        forced = (
            best_tactical_move(board, return_reason=True)
            if self.tactical_guard
            else {"move": None, "reason": None}
        )
        forced_move = forced["move"]
        if forced_move is not None:
            move_probs[forced_move] = 1.0
            self.forced_tactical_moves += 1
            if forced["reason"] == "forcing_win":
                self.threat_solver_moves += 1
            return (forced_move, move_probs) if return_prob else forced_move

        action_probs, _root_value = self.policy_value_net.policy_value_fn(board)
        policy_by_move = {move: float(prob) for move, prob in action_probs}
        ranked = ranked_tactical_moves(board)
        tactical_by_move = {item["move"]: item for item in ranked}
        candidates = []
        seen = set()
        for item in ranked[: self.beam_width]:
            candidates.append(item["move"])
            seen.add(item["move"])
        policy_top = sorted(
            policy_by_move.items(),
            key=lambda item: item[1],
            reverse=True,
        )[: self.policy_top_k]
        for move, _prob in policy_top:
            if move not in seen:
                candidates.append(move)
                seen.add(move)

        current_player = board.get_current_player()
        scored = []
        for move in candidates:
            value_after_move, opponent_reply = self._candidate_value(
                board,
                move,
                current_player,
            )
            self.candidate_evaluations += 1
            tactical_score = max(0.0, tactical_by_move.get(move, {}).get("score", 0.0))
            tactical_norm = np.log1p(tactical_score) / np.log1p(WIN_SCORE)
            fork_count = fork_threat_count(
                board,
                move,
                current_player,
                threshold=self.fork_threshold,
            )
            policy_prior = max(policy_by_move.get(move, 0.0), 1e-12)
            score = (
                self.value_weight * value_after_move
                + self.tactical_weight * tactical_norm
                + self.fork_weight * fork_count
                + self.policy_weight * np.log(policy_prior)
            )
            if opponent_reply:
                score -= self.reply_penalty
            scored.append((move, score))

        if not scored:
            move = int(self._rng.choice(list(board.availables)))
            move_probs[move] = 1.0
            self.beam_moves += 1
            return (move, move_probs) if return_prob else move

        acts = np.array([move for move, _score in scored], dtype=np.int64)
        scores = np.array([score for _move, score in scored], dtype=np.float64)
        if temp <= 1e-3:
            probs = np.zeros(len(acts), dtype=np.float64)
            probs[int(np.argmax(scores))] = 1.0
        else:
            adjusted = scores / max(float(temp), 1e-6)
            adjusted -= np.max(adjusted)
            probs = np.exp(adjusted)
            probs_sum = float(np.sum(probs))
            if not np.isfinite(probs_sum) or probs_sum <= 0:
                probs = np.full(len(acts), 1.0 / len(acts))
            else:
                probs = probs / probs_sum

            if self.noise_frac > 0 and len(acts) > 1:
                alpha = max(float(self.dirichlet_alpha), 1e-6)
                noise = self._rng.dirichlet(np.full(len(acts), alpha))
                probs = (1.0 - self.noise_frac) * probs + self.noise_frac * noise
                probs = probs / np.sum(probs)

        move = int(self._rng.choice(acts, p=probs))
        move_probs[acts] = probs
        self.beam_moves += 1
        if fork_threat_count(
            board,
            move,
            board.get_current_player(),
            threshold=self.fork_threshold,
        ) >= 2:
            self.fork_moves += 1
        return (move, move_probs) if return_prob else move


def _new_self_play_player(policy_value_net, config, seed):
    mode = config.get("self_play_mode", "mcts")
    if mode == "mcts":
        return MCTSPlayer(
            policy_value_net.policy_value_fn,
            c_puct=config["c_puct"],
            n_playout=config["n_playout"],
            is_selfplay=1,
            use_parallel=False,
            tactical_threshold=config.get("mcts_tactical_threshold"),
            two_ply_threats=config.get("mcts_two_ply_threats", False),
            two_ply_max_candidates=config.get("mcts_two_ply_max_candidates", 16),
            two_ply_max_replies=config.get("mcts_two_ply_max_replies", 6),
            two_ply_max_followups=config.get("mcts_two_ply_max_followups", 12),
            dirichlet_alpha=config.get("mcts_dirichlet_alpha", 0.3),
            dirichlet_frac=config.get("mcts_dirichlet_frac", 0.25),
            dirichlet_moves=config.get("mcts_dirichlet_moves"),
            tactical_prior_weight=config.get("mcts_tactical_prior_weight", 0.0),
            tactical_prior_temperature=config.get("mcts_tactical_prior_temperature", 1.0),
            tactical_prior_two_ply_bonus=config.get("mcts_tactical_prior_two_ply_bonus", 0.0),
            tactical_prior_two_ply_max_candidates=config.get("mcts_tactical_prior_two_ply_max_candidates", 16),
            tactical_prior_two_ply_max_replies=config.get("mcts_tactical_prior_two_ply_max_replies", 6),
            tactical_prior_two_ply_max_followups=config.get("mcts_tactical_prior_two_ply_max_followups", 12),
            tactical_leaf_eval=config.get("mcts_tactical_leaf_eval", False),
            tactical_leaf_win_value=config.get("mcts_tactical_leaf_win_value", 1.0),
            tactical_leaf_loss_value=config.get("mcts_tactical_leaf_loss_value", 0.95),
            tactical_leaf_forcing_value=config.get("mcts_tactical_leaf_forcing_value", 0.85),
            tactical_leaf_two_ply_value=config.get("mcts_tactical_leaf_two_ply_value", 0.70),
            tactical_leaf_two_ply=config.get("mcts_tactical_leaf_two_ply", False),
            tactical_leaf_max_candidates=config.get("mcts_tactical_leaf_max_candidates", 16),
            tactical_leaf_max_replies=config.get("mcts_tactical_leaf_max_replies", 6),
            tactical_leaf_max_followups=config.get("mcts_tactical_leaf_max_followups", 12),
        )
    if mode == "policy_tactical":
        return PolicyTacticalSelfPlayPlayer(
            policy_value_net,
            tactical_guard=config.get("policy_tactical_guard", True),
            dirichlet_alpha=config.get("policy_dirichlet_alpha", 0.3),
            noise_frac=config.get("policy_noise_frac", 0.15),
            seed=seed,
        )
    if mode == "tactical_beam":
        return TacticalBeamSelfPlayPlayer(
            policy_value_net,
            tactical_guard=config.get("policy_tactical_guard", True),
            beam_width=config.get("beam_width", 8),
            policy_top_k=config.get("beam_policy_top_k", 8),
            value_weight=config.get("beam_value_weight", 1.2),
            policy_weight=config.get("beam_policy_weight", 0.45),
            tactical_weight=config.get("beam_tactical_weight", 1.0),
            reply_penalty=config.get("beam_reply_penalty", 0.35),
            fork_weight=config.get("beam_fork_weight", 0.0),
            fork_threshold=config.get("beam_fork_threshold", OPEN_FOUR_SCORE),
            dirichlet_alpha=config.get("policy_dirichlet_alpha", 0.3),
            noise_frac=config.get("policy_noise_frac", 0.10),
            seed=seed,
        )
    raise ValueError(f"Unknown self_play_mode: {mode}")


def _self_play_temperature(config):
    cutoff = config.get("self_play_temp_cutoff")
    if cutoff is None:
        return config["temp"]

    early_temp = float(config.get("temp", 1.0))
    late_temp = float(config.get("self_play_late_temp", 1e-3))
    cutoff = max(0, int(cutoff))

    def schedule(move_idx, _board):
        return early_temp if move_idx < cutoff else late_temp

    return schedule


def _self_play_stats(player, mode):
    stats = {"self_play_mode": mode}
    if hasattr(player, "forced_tactical_moves"):
        stats["forced_tactical_moves"] = int(player.forced_tactical_moves)
    if hasattr(player, "threat_solver_moves"):
        stats["threat_solver_moves"] = int(player.threat_solver_moves)
    if hasattr(player, "two_ply_threat_moves"):
        stats["two_ply_threat_moves"] = int(player.two_ply_threat_moves)
    if hasattr(player, "policy_moves"):
        stats["policy_moves"] = int(player.policy_moves)
    if hasattr(player, "search_moves"):
        stats["search_moves"] = int(player.search_moves)
    if hasattr(player, "dirichlet_noise_moves"):
        stats["dirichlet_noise_moves"] = int(player.dirichlet_noise_moves)
    if hasattr(player, "no_noise_moves"):
        stats["no_noise_moves"] = int(player.no_noise_moves)
    if hasattr(player, "tactical_prior_searches"):
        stats["tactical_prior_searches"] = int(player.tactical_prior_searches)
    if hasattr(player, "tactical_prior_two_ply_applications"):
        stats["tactical_prior_two_ply_hits"] = int(player.tactical_prior_two_ply_applications)
    if hasattr(player, "tactical_leaf_evaluations"):
        stats["tactical_leaf_evaluations"] = int(player.tactical_leaf_evaluations)
    if hasattr(player, "tactical_leaf_positive"):
        stats["tactical_leaf_positive"] = int(player.tactical_leaf_positive)
    if hasattr(player, "tactical_leaf_negative"):
        stats["tactical_leaf_negative"] = int(player.tactical_leaf_negative)
    if hasattr(player, "tactical_leaf_reasons"):
        for reason, count in player.tactical_leaf_reasons.items():
            stats[f"tactical_leaf_{reason}"] = int(count)
    if hasattr(player, "beam_moves"):
        stats["beam_moves"] = int(player.beam_moves)
    if hasattr(player, "fork_moves"):
        stats["fork_moves"] = int(player.fork_moves)
    if hasattr(player, "candidate_evaluations"):
        stats["candidate_evaluations"] = int(player.candidate_evaluations)
    return stats


def _heuristic_play_data(config, seed):
    board = _new_board(config)
    board.init_board()
    teacher_player = board.players[seed % len(board.players)]
    players = {
        teacher_player: HeuristicPlayer(seed=seed),
        board.players[0] if teacher_player == board.players[1] else board.players[1]: RandomPlayer(seed=seed + 1),
    }
    for player_index, player in players.items():
        player.set_player_ind(player_index)

    states = []
    move_probs = []
    current_players = []
    moves = []

    while True:
        current_player = board.get_current_player()
        player = players[current_player]
        move = player.get_action(board)
        if move is None:
            winner = -1
            break

        if current_player == teacher_player:
            policy_target = np.zeros(board.width * board.height)
            policy_target[move] = 1.0
            states.append(board.current_state())
            move_probs.append(policy_target)
            current_players.append(current_player)
        moves.append(move)

        board.do_move(move)
        end, winner = board.game_end()
        if end:
            break

    winners_z = np.zeros(len(current_players))
    if winner != -1:
        players_array = np.array(current_players)
        winners_z[players_array == winner] = 1.0
        winners_z[players_array != winner] = -1.0

    return winner, zip(states, move_probs, winners_z), moves


def _line_start_bounds(width, height, n_in_row, dx, dy):
    min_x = max(0, -(n_in_row - 1) * dx)
    max_x = min(width - 1, width - 1 - (n_in_row - 1) * dx)
    min_y = max(0, -(n_in_row - 1) * dy)
    max_y = min(height - 1, height - 1 - (n_in_row - 1) * dy)
    return min_x, max_x, min_y, max_y


def _place_stone(board, move, player):
    board.states[move] = player
    if move in board.availables:
        board.availables.remove(move)
    board.last_move = move


TACTICAL_PUZZLE_FOCUS_PATTERNS = {
    "balanced": (0, 1, 2, 3),
    "win_conversion": (0, 2, 0, 2, 1, 3, 0, 2),
}


def _tactical_puzzle_kind(config, puzzle_idx):
    focus = config.get("tactical_puzzle_focus", "balanced")
    pattern = TACTICAL_PUZZLE_FOCUS_PATTERNS.get(focus)
    if pattern is None:
        pattern = TACTICAL_PUZZLE_FOCUS_PATTERNS["balanced"]
    return pattern[puzzle_idx % len(pattern)]


def _tactical_puzzle_sample(config, rng, puzzle_idx):
    board = _new_board(config)
    board.init_board()
    current_player = board.get_current_player()
    opponent = board.players[0] if current_player == board.players[1] else board.players[1]
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    dx, dy = directions[puzzle_idx % len(directions)]
    min_x, max_x, min_y, max_y = _line_start_bounds(
        board.width,
        board.height,
        board.n_in_row,
        dx,
        dy,
    )
    start_x = rng.randint(min_x, max_x)
    start_y = rng.randint(min_y, max_y)
    line = [
        (start_x + step * dx, start_y + step * dy)
        for step in range(board.n_in_row)
    ]
    gap_idx = rng.randrange(board.n_in_row)
    target_x, target_y = line[gap_idx]
    target_move = target_y * board.width + target_x

    pattern_kind = _tactical_puzzle_kind(config, puzzle_idx)
    pattern_player = current_player if pattern_kind in (0, 2) else opponent
    target_count_after_move = board.n_in_row if pattern_kind in (0, 1) else board.n_in_row - 1
    stones_to_place = target_count_after_move - 1
    fill_candidates = [idx for idx in range(board.n_in_row) if idx != gap_idx]
    fill_candidates.sort(key=lambda idx: (abs(idx - gap_idx), rng.random()))
    fill_indices = set(fill_candidates[:stones_to_place])
    for idx, (x, y) in enumerate(line):
        if idx == gap_idx or idx not in fill_indices:
            continue
        _place_stone(board, y * board.width + x, pattern_player)

    forbidden = {target_move}
    forbidden.update(y * board.width + x for x, y in line)
    candidate_moves = [move for move in board.availables if move not in forbidden]
    rng.shuffle(candidate_moves)
    for move in candidate_moves[: rng.randint(0, 3)]:
        player = current_player if rng.random() < 0.5 else opponent
        _place_stone(board, move, player)

    board.current_player = current_player
    policy_target = np.zeros(board.width * board.height)
    policy_target[target_move] = 1.0
    if pattern_player == current_player:
        winner_target = (
            config.get("tactical_win_value", 1.0)
            if target_count_after_move >= board.n_in_row
            else config.get("tactical_attack_threat_value", 0.75)
        )
    else:
        winner_target = (
            config.get("tactical_block_win_value", 0.0)
            if target_count_after_move >= board.n_in_row
            else config.get("tactical_block_threat_value", 0.25)
        )
    return board.current_state(), policy_target, float(winner_target)


def _tactical_puzzle_data(config, seed, count):
    rng = random.Random(seed)
    return [
        _tactical_puzzle_sample(config, rng, puzzle_idx)
        for puzzle_idx in range(count)
    ]


def _coord_to_move(board, x, y):
    return y * board.width + x


def _inside_board(board, x, y):
    return 0 <= x < board.width and 0 <= y < board.height


def _fork_threat_puzzle_board(config, rng, puzzle_idx):
    board = _new_board(config)
    board.init_board()
    current_player = board.get_current_player()
    direction_pairs = (
        (DIRECTIONS[0], DIRECTIONS[1]),
        (DIRECTIONS[0], DIRECTIONS[2]),
        (DIRECTIONS[0], DIRECTIONS[3]),
        (DIRECTIONS[1], DIRECTIONS[2]),
        (DIRECTIONS[1], DIRECTIONS[3]),
        (DIRECTIONS[2], DIRECTIONS[3]),
    )
    first_pair = puzzle_idx % len(direction_pairs)
    pairs = list(direction_pairs[first_pair:]) + list(direction_pairs[:first_pair])

    target_x = target_y = None
    chosen_pair = None
    for pair in pairs:
        for _attempt in range(200):
            candidate_x = rng.randrange(board.width)
            candidate_y = rng.randrange(board.height)
            required = []
            for dx, dy in pair:
                required.extend(
                    (candidate_x + step * dx, candidate_y + step * dy)
                    for step in (-1, 1, 2, 3, 4)
                )
            if all(_inside_board(board, x, y) for x, y in required):
                target_x = candidate_x
                target_y = candidate_y
                chosen_pair = pair
                break
        if chosen_pair is not None:
            break

    if chosen_pair is None:
        return _new_board(config), None, ()

    target_move = _coord_to_move(board, target_x, target_y)
    protected = {target_move}
    for dx, dy in chosen_pair:
        for step in (-1, 1, 2, 3, 4):
            protected.add(_coord_to_move(board, target_x + step * dx, target_y + step * dy))
        for step in (1, 2, 3):
            _place_stone(
                board,
                _coord_to_move(board, target_x + step * dx, target_y + step * dy),
                current_player,
            )

    noise_stones = int(config.get("hard_position_noise_stones", 4) or 0)
    candidate_moves = [move for move in board.availables if move not in protected]
    rng.shuffle(candidate_moves)
    opponent = board.players[0] if current_player == board.players[1] else board.players[1]
    for move in candidate_moves[:noise_stones]:
        player = current_player if rng.random() < 0.35 else opponent
        _place_stone(board, move, player)

    board.current_player = current_player
    return board, target_move, chosen_pair


def _hard_position_puzzle_sample(config, rng, puzzle_idx):
    board, target_move, _directions = _fork_threat_puzzle_board(config, rng, puzzle_idx)
    if target_move is None:
        return _tactical_puzzle_sample(config, rng, puzzle_idx)

    policy_target = np.zeros(board.width * board.height)
    policy_target[target_move] = 1.0
    winner_target = float(config.get("hard_position_fork_value", 0.95))
    return board.current_state(), policy_target, winner_target


def _hard_position_puzzle_data(config, seed, count):
    rng = random.Random(seed)
    return [
        _hard_position_puzzle_sample(config, rng, puzzle_idx)
        for puzzle_idx in range(count)
    ]


def _threat_space_puzzle_board(config, rng, puzzle_idx):
    board = _new_board(config)
    board.init_board()
    current_player = board.get_current_player()
    opponent = board.players[0] if current_player == board.players[1] else board.players[1]
    direction_pairs = (
        (DIRECTIONS[0], DIRECTIONS[1]),
        (DIRECTIONS[0], DIRECTIONS[2]),
        (DIRECTIONS[0], DIRECTIONS[3]),
        (DIRECTIONS[1], DIRECTIONS[2]),
        (DIRECTIONS[1], DIRECTIONS[3]),
        (DIRECTIONS[2], DIRECTIONS[3]),
    )
    first_pair = puzzle_idx % len(direction_pairs)
    pairs = list(direction_pairs[first_pair:]) + list(direction_pairs[:first_pair])
    reply_limit = int(config.get("threat_space_reply_limit", 8) or 0)
    followup_limit = int(config.get("threat_space_followup_limit", 16) or 0)
    validate_puzzles = bool(config.get("threat_space_validate_puzzles", False))

    for pair in pairs:
        for _attempt in range(200):
            board = _new_board(config)
            board.init_board()
            current_player = board.get_current_player()
            opponent = board.players[0] if current_player == board.players[1] else board.players[1]
            target_x = rng.randrange(board.width)
            target_y = rng.randrange(board.height)
            required = []
            for dx, dy in pair:
                required.extend(
                    (target_x + step * dx, target_y + step * dy)
                    for step in (-2, -1, 1, 2)
                )
            if not all(_inside_board(board, x, y) for x, y in required):
                continue

            target_move = _coord_to_move(board, target_x, target_y)
            protected = {target_move}
            for dx, dy in pair:
                for step in (-2, -1, 1, 2):
                    protected.add(_coord_to_move(board, target_x + step * dx, target_y + step * dy))
                for step in (-1, 1):
                    _place_stone(
                        board,
                        _coord_to_move(board, target_x + step * dx, target_y + step * dy),
                        current_player,
                    )

            noise_stones = int(config.get("threat_space_noise_stones", 4) or 0)
            candidate_moves = [move for move in board.availables if move not in protected]
            rng.shuffle(candidate_moves)
            for move in candidate_moves[:noise_stones]:
                player = current_player if rng.random() < 0.35 else opponent
                _place_stone(board, move, player)

            board.current_player = current_player
            if not validate_puzzles or creates_bounded_two_ply_threat(
                board,
                target_move,
                current_player,
                max_replies=reply_limit,
                max_followups=followup_limit,
                include_one_ply=False,
            ):
                return board, target_move, pair

    return _new_board(config), None, ()


def _threat_space_puzzle_sample(config, rng, puzzle_idx):
    board, target_move, _directions = _threat_space_puzzle_board(config, rng, puzzle_idx)
    if target_move is None:
        return _hard_position_puzzle_sample(config, rng, puzzle_idx)

    policy_target = np.zeros(board.width * board.height)
    policy_target[target_move] = 1.0
    winner_target = float(config.get("threat_space_value", 0.85))
    return board.current_state(), policy_target, winner_target


def _threat_space_puzzle_data(config, seed, count):
    rng = random.Random(seed)
    return [
        _threat_space_puzzle_sample(config, rng, puzzle_idx)
        for puzzle_idx in range(count)
    ]


def _policy_target_for_move(board, move=None):
    policy_target = np.zeros(board.width * board.height)
    if move is None:
        available = list(board.availables)
        if available:
            policy_target[available] = 1.0 / len(available)
    else:
        policy_target[int(move)] = 1.0
    return policy_target


def _copy_after_move(board, move, player=None):
    child = copy.deepcopy(board)
    if player is not None:
        child.current_player = player
    child.do_move(move)
    return child


def _proof_followup_move(board, attacker, max_followups):
    attacker_wins = winning_moves(board, attacker)
    if attacker_wins:
        return attacker_wins[0]
    return best_forcing_win_move(
        board,
        attacker,
        max_candidates=max_followups,
    )


def _threat_space_proof_value_data(config, seed, count):
    rng = random.Random(seed)
    data = []
    roots = 0
    defender_states = 0
    replies_considered = 0
    followup_states = 0
    skipped = 0
    reply_limit = int(config.get("threat_space_proof_reply_limit", 4) or 0)
    followup_limit = int(config.get("threat_space_proof_followup_limit", 12) or 0)
    root_value = float(config.get("threat_space_proof_root_value", config.get("threat_space_value", 0.85)))
    defender_value = -abs(float(config.get("threat_space_proof_defender_value", root_value)))
    followup_value = float(config.get("threat_space_proof_followup_value", 0.95))

    for puzzle_idx in range(count):
        board, target_move, _directions = _threat_space_puzzle_board(config, rng, puzzle_idx)
        if target_move is None:
            skipped += 1
            continue
        attacker = board.get_current_player()
        if not creates_bounded_two_ply_threat(
            board,
            target_move,
            attacker,
            max_replies=config.get("threat_space_reply_limit", 8),
            max_followups=config.get("threat_space_followup_limit", 16),
            include_one_ply=False,
        ):
            skipped += 1
            continue

        data.append((board.current_state(), _policy_target_for_move(board, target_move), root_value))
        roots += 1

        child = _copy_after_move(board, target_move, attacker)
        defender = child.get_current_player()
        data.append((child.current_state(), _policy_target_for_move(child), defender_value))
        defender_states += 1

        replies = plausible_reply_moves(
            child,
            attacker,
            defender=defender,
            max_replies=reply_limit,
        )
        for reply in replies:
            replies_considered += 1
            reply_board = _copy_after_move(child, reply, defender)
            end, winner = reply_board.game_end()
            if end and winner != attacker:
                continue
            followup = _proof_followup_move(
                reply_board,
                attacker,
                max_followups=followup_limit,
            )
            if followup is None:
                continue
            data.append((
                reply_board.current_state(),
                _policy_target_for_move(reply_board, followup),
                followup_value,
            ))
            followup_states += 1

    return data, {
        "threat_space_proof_samples": len(data),
        "threat_space_proof_positions": count,
        "threat_space_proof_roots": roots,
        "threat_space_proof_defender_states": defender_states,
        "threat_space_proof_replies": replies_considered,
        "threat_space_proof_followups": followup_states,
        "threat_space_proof_skipped": skipped,
    }


def _mcts_distill_position_board(config, rng, puzzle_idx, source_override=None):
    source = source_override or config.get("mcts_distill_source", "mixed")
    if source not in {"mixed", "hard_position", "threat_space"}:
        raise ValueError(f"Unknown mcts_distill_source: {source}")

    use_threat_space = source == "threat_space" or (
        source == "mixed" and puzzle_idx % 2 == 1
    )
    if use_threat_space:
        board, target_move, _directions = _threat_space_puzzle_board(config, rng, puzzle_idx)
        value = float(config.get("mcts_distill_threat_space_value", config.get("threat_space_value", 0.85)))
        source_name = "threat_space"
        if target_move is not None:
            return board, target_move, value, source_name

    board, target_move, _directions = _fork_threat_puzzle_board(config, rng, puzzle_idx)
    value = float(config.get("mcts_distill_hard_position_value", config.get("hard_position_fork_value", 0.95)))
    source_name = "hard_position"
    if target_move is None:
        if not hasattr(board, "availables"):
            board.init_board()
        return board, None, value, "fallback_empty"
    return board, target_move, value, source_name


def _mcts_distill_source_config(config, source):
    overrides = config.get("mcts_distill_source_overrides", {}) or {}
    source_override = overrides.get(source, {}) or {}
    if not source_override:
        return config
    source_config = dict(config)
    source_config.update(source_override)
    return source_config


def _mcts_distill_source_plan(config, count):
    source_counts = config.get("mcts_distill_source_counts")
    if not source_counts:
        return [(config.get("mcts_distill_source", "mixed"), count)]
    plan = []
    for source in ("hard_position", "threat_space"):
        source_count = int(source_counts.get(source, 0) or 0)
        if source_count > 0:
            plan.append((source, source_count))
    extra_sources = sorted(
        source
        for source in source_counts
        if source not in {"hard_position", "threat_space"}
    )
    for source in extra_sources:
        source_count = int(source_counts.get(source, 0) or 0)
        if source_count > 0:
            plan.append((source, source_count))
    return plan


def _mcts_distill_policy_target(policy_value_net, board, target_move, config):
    mcts = MCTS(
        policy_value_net.policy_value_fn,
        c_puct=config.get("mcts_distill_c_puct", config.get("c_puct", 5)),
        n_playout=max(1, int(config.get("mcts_distill_playouts", 16) or 1)),
        tactical_prior_weight=config.get("mcts_distill_tactical_prior_weight", 0.0),
        tactical_prior_temperature=config.get("mcts_distill_tactical_prior_temperature", 1.0),
        tactical_prior_two_ply_bonus=config.get("mcts_distill_tactical_prior_two_ply_bonus", 0.0),
        tactical_prior_two_ply_max_candidates=config.get("mcts_distill_tactical_prior_two_ply_max_candidates", 16),
        tactical_prior_two_ply_max_replies=config.get("mcts_distill_tactical_prior_two_ply_max_replies", 6),
        tactical_prior_two_ply_max_followups=config.get("mcts_distill_tactical_prior_two_ply_max_followups", 12),
        tactical_leaf_eval=config.get("mcts_distill_tactical_leaf_eval", False),
        tactical_leaf_win_value=config.get("mcts_distill_tactical_leaf_win_value", 1.0),
        tactical_leaf_loss_value=config.get("mcts_distill_tactical_leaf_loss_value", 0.95),
        tactical_leaf_forcing_value=config.get("mcts_distill_tactical_leaf_forcing_value", 0.85),
        tactical_leaf_two_ply_value=config.get("mcts_distill_tactical_leaf_two_ply_value", 0.70),
        tactical_leaf_two_ply=config.get("mcts_distill_tactical_leaf_two_ply", False),
        tactical_leaf_max_candidates=config.get("mcts_distill_tactical_leaf_max_candidates", 16),
        tactical_leaf_max_replies=config.get("mcts_distill_tactical_leaf_max_replies", 6),
        tactical_leaf_max_followups=config.get("mcts_distill_tactical_leaf_max_followups", 12),
    )
    acts, probs = mcts.get_move_probs(
        board,
        temp=max(float(config.get("mcts_distill_temp", 1.0)), 1e-6),
    )
    policy_target = np.zeros(board.width * board.height, dtype=np.float64)
    if acts:
        policy_target[list(acts)] = probs

    policy_sum = float(np.sum(policy_target))
    if policy_sum <= 0.0 or not np.isfinite(policy_sum):
        if target_move is not None:
            policy_target[int(target_move)] = 1.0
        else:
            available = list(board.availables)
            policy_target[available] = 1.0 / len(available)
    else:
        policy_target = policy_target / policy_sum

    search_policy = policy_target.copy()
    search_target_mass = 0.0 if target_move is None else float(search_policy[int(target_move)])
    search_top_move = int(np.argmax(search_policy))
    search_top_hit = bool(target_move is not None and search_top_move == int(target_move))
    min_target_mass = float(config.get("mcts_distill_min_target_mass", 0.0) or 0.0)
    require_target_top = bool(config.get("mcts_distill_require_target_top", False))
    accepted = target_move is not None or not (min_target_mass > 0.0 or require_target_top)
    if target_move is not None:
        accepted = True
        if require_target_top and not search_top_hit:
            accepted = False
        if min_target_mass > 0.0 and search_target_mass < min_target_mass:
            accepted = False

    base_stats = {
        "search_target_mass": search_target_mass,
        "search_target_top_hit": search_top_hit,
        "accepted": accepted,
        "tactical_leaf_evaluations": int(mcts.tactical_leaf_evaluations),
        "tactical_leaf_positive": int(mcts.tactical_leaf_positive),
        "tactical_leaf_negative": int(mcts.tactical_leaf_negative),
    }
    if not accepted:
        return None, {
            **base_stats,
            "target_mass": search_target_mass,
            "target_top_hit": search_top_hit,
            "entropy": 0.0,
        }

    target_mode = config.get("mcts_distill_target_mode", "visits")
    if target_mode not in {"visits", "search_top", "teacher"}:
        raise ValueError(f"Unknown mcts_distill_target_mode: {target_mode}")

    if target_mode == "search_top":
        policy_target = np.zeros_like(search_policy)
        policy_target[search_top_move] = 1.0
    elif target_mode == "teacher" and target_move is not None:
        policy_target = np.zeros_like(search_policy)
        policy_target[int(target_move)] = 1.0
    else:
        target_blend = max(0.0, min(1.0, float(config.get("mcts_distill_target_blend", 0.0) or 0.0)))
        if target_move is not None and target_blend > 0.0:
            one_hot = np.zeros_like(policy_target)
            one_hot[int(target_move)] = 1.0
            policy_target = (1.0 - target_blend) * policy_target + target_blend * one_hot

        policy_power = max(float(config.get("mcts_distill_policy_power", 1.0) or 1.0), 1e-6)
        if policy_power != 1.0:
            positive = policy_target > 0.0
            policy_target[positive] = np.power(policy_target[positive], policy_power)

    policy_sum = float(np.sum(policy_target))
    if policy_sum <= 0.0 or not np.isfinite(policy_sum):
        policy_target = search_policy
        policy_sum = float(np.sum(policy_target))
    policy_target = policy_target / max(policy_sum, 1e-12)

    target_mass = 0.0 if target_move is None else float(policy_target[int(target_move)])
    top_move = int(np.argmax(policy_target))
    nonzero = policy_target[policy_target > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero))) if len(nonzero) else 0.0
    return policy_target, {
        **base_stats,
        "target_mass": target_mass,
        "target_top_hit": bool(target_move is not None and top_move == int(target_move)),
        "entropy": entropy,
    }


def _mcts_distill_data(policy_value_net, config, seed, count):
    rng = random.Random(seed)
    data = []
    source_counts = {}
    source_stats = {}
    target_mass_sum = 0.0
    search_target_mass_sum = 0.0
    entropy_sum = 0.0
    target_top_hits = 0
    search_target_top_hits = 0
    leaf_evaluations = 0
    leaf_positive = 0
    leaf_negative = 0
    attempts = 0
    skipped = 0

    for requested_source, requested_count in _mcts_distill_source_plan(config, count):
        source_config = _mcts_distill_source_config(config, requested_source)
        max_attempts_factor = max(
            1.0,
            float(source_config.get("mcts_distill_max_attempts_factor", 1.0) or 1.0),
        )
        max_attempts = max(requested_count, int(np.ceil(requested_count * max_attempts_factor)))
        source_attempts = 0
        source_skipped = 0
        source_samples = 0
        source_target_mass_sum = 0.0
        source_search_mass_sum = 0.0
        source_top_hits = 0
        source_search_top_hits = 0
        source_entropy_sum = 0.0
        source_leaf_evaluations = 0
        source_leaf_positive = 0
        source_leaf_negative = 0

        while source_samples < requested_count and source_attempts < max_attempts:
            puzzle_idx = source_attempts
            source_attempts += 1
            attempts += 1
            board, target_move, value, source = _mcts_distill_position_board(
                source_config,
                rng,
                puzzle_idx,
                source_override=requested_source,
            )
            policy_target, stats = _mcts_distill_policy_target(
                policy_value_net,
                board,
                target_move,
                source_config,
            )
            if not stats["accepted"] or policy_target is None:
                skipped += 1
                source_skipped += 1
                continue
            data.append((board.current_state(), policy_target, value))
            source_samples += 1
            source_counts[source] = source_counts.get(source, 0) + 1
            target_mass_sum += stats["target_mass"]
            search_target_mass_sum += stats["search_target_mass"]
            entropy_sum += stats["entropy"]
            target_top_hits += int(stats["target_top_hit"])
            search_target_top_hits += int(stats["search_target_top_hit"])
            leaf_evaluations += stats["tactical_leaf_evaluations"]
            leaf_positive += stats["tactical_leaf_positive"]
            leaf_negative += stats["tactical_leaf_negative"]
            source_target_mass_sum += stats["target_mass"]
            source_search_mass_sum += stats["search_target_mass"]
            source_entropy_sum += stats["entropy"]
            source_top_hits += int(stats["target_top_hit"])
            source_search_top_hits += int(stats["search_target_top_hit"])
            source_leaf_evaluations += stats["tactical_leaf_evaluations"]
            source_leaf_positive += stats["tactical_leaf_positive"]
            source_leaf_negative += stats["tactical_leaf_negative"]

        source_divisor = max(1, source_samples)
        source_stats[requested_source] = {
            "requested": requested_count,
            "samples": source_samples,
            "attempts": source_attempts,
            "skipped": source_skipped,
            "accept_rate": source_samples / max(1, source_attempts),
            "target_mass": source_target_mass_sum / source_divisor,
            "target_top_rate": source_top_hits / source_divisor,
            "search_target_mass": source_search_mass_sum / source_divisor,
            "search_top_rate": source_search_top_hits / source_divisor,
            "target_entropy": source_entropy_sum / source_divisor,
            "leaf_evaluations": source_leaf_evaluations,
            "leaf_positive": source_leaf_positive,
            "leaf_negative": source_leaf_negative,
        }

    divisor = max(1, len(data))
    return data, {
        "mcts_distill_samples": len(data),
        "mcts_distill_attempts": attempts,
        "mcts_distill_skipped": skipped,
        "mcts_distill_accept_rate": len(data) / max(1, attempts),
        "mcts_distill_target_mass": target_mass_sum / divisor,
        "mcts_distill_target_top_rate": target_top_hits / divisor,
        "mcts_distill_search_target_mass": search_target_mass_sum / divisor,
        "mcts_distill_search_top_rate": search_target_top_hits / divisor,
        "mcts_distill_entropy": entropy_sum / divisor,
        "mcts_distill_leaf_evaluations": leaf_evaluations,
        "mcts_distill_leaf_positive": leaf_positive,
        "mcts_distill_leaf_negative": leaf_negative,
        "mcts_distill_sources": source_counts,
        "mcts_distill_source_stats": source_stats,
    }


def _apply_self_play_draw_value(play_data, winner, draw_value):
    draw_value = float(draw_value or 0.0)
    if winner != -1 or draw_value == 0.0:
        return play_data
    return [
        (state, move_probs, draw_value)
        for state, move_probs, _winner_z in play_data
    ]


def _anchor_distillation_data(anchor_policy, data_buffer, config, seed, count):
    if count <= 0 or not data_buffer:
        return []

    rng = random.Random(seed)
    samples = rng.sample(list(data_buffer), min(count, len(data_buffer)))
    states = [sample[0] for sample in samples]
    act_probs, values = anchor_policy.policy_value(states)
    return [
        (state, act_probs[idx], float(values[idx][0]))
        for idx, state in enumerate(states)
    ]


def _checkpoint_name(preset, tag, games_trained):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"{preset}_{tag}_g{games_trained}_{stamp}.model"


def _public_config(config):
    keys = [
        "preset",
        "board_width",
        "board_height",
        "n_in_row",
        "num_res_blocks",
        "num_filters",
        "architecture",
        "self_play_mode",
        "self_play_games",
        "n_playout",
        "eval_n_playout",
        "mcts_tactical_threshold",
        "mcts_two_ply_threats",
        "mcts_two_ply_max_candidates",
        "mcts_two_ply_max_replies",
        "mcts_two_ply_max_followups",
        "mcts_dirichlet_alpha",
        "mcts_dirichlet_frac",
        "mcts_dirichlet_moves",
        "mcts_tactical_prior_weight",
        "mcts_tactical_prior_temperature",
        "mcts_tactical_prior_two_ply_bonus",
        "mcts_tactical_prior_two_ply_max_candidates",
        "mcts_tactical_prior_two_ply_max_replies",
        "mcts_tactical_prior_two_ply_max_followups",
        "mcts_tactical_leaf_eval",
        "mcts_tactical_leaf_win_value",
        "mcts_tactical_leaf_loss_value",
        "mcts_tactical_leaf_forcing_value",
        "mcts_tactical_leaf_two_ply_value",
        "mcts_tactical_leaf_two_ply",
        "mcts_tactical_leaf_max_candidates",
        "mcts_tactical_leaf_max_replies",
        "mcts_tactical_leaf_max_followups",
        "eval_games",
        "batch_size",
        "epochs",
        "expert_games",
        "tactical_puzzles",
        "tactical_puzzle_focus",
        "hard_position_puzzles",
        "hard_position_fork_value",
        "hard_position_noise_stones",
        "threat_space_puzzles",
        "threat_space_value",
        "threat_space_noise_stones",
        "threat_space_reply_limit",
        "threat_space_followup_limit",
        "threat_space_validate_puzzles",
        "threat_space_proof_positions",
        "threat_space_proof_reply_limit",
        "threat_space_proof_followup_limit",
        "threat_space_proof_root_value",
        "threat_space_proof_defender_value",
        "threat_space_proof_followup_value",
        "threat_space_proof_batch_size",
        "threat_space_proof_epochs",
        "threat_space_proof_policy_loss_weight",
        "threat_space_proof_value_loss_weight",
        "threat_space_proof_add_to_replay",
        "mcts_distill_positions",
        "mcts_distill_source",
        "mcts_distill_source_counts",
        "mcts_distill_source_overrides",
        "mcts_distill_playouts",
        "mcts_distill_temp",
        "mcts_distill_c_puct",
        "mcts_distill_batch_size",
        "mcts_distill_epochs",
        "mcts_distill_policy_loss_weight",
        "mcts_distill_value_loss_weight",
        "mcts_distill_target_mode",
        "mcts_distill_target_blend",
        "mcts_distill_policy_power",
        "mcts_distill_require_target_top",
        "mcts_distill_min_target_mass",
        "mcts_distill_max_attempts_factor",
        "mcts_distill_hard_position_value",
        "mcts_distill_threat_space_value",
        "mcts_distill_tactical_prior_weight",
        "mcts_distill_tactical_prior_temperature",
        "mcts_distill_tactical_prior_two_ply_bonus",
        "mcts_distill_tactical_prior_two_ply_max_candidates",
        "mcts_distill_tactical_prior_two_ply_max_replies",
        "mcts_distill_tactical_prior_two_ply_max_followups",
        "mcts_distill_tactical_leaf_eval",
        "mcts_distill_tactical_leaf_win_value",
        "mcts_distill_tactical_leaf_loss_value",
        "mcts_distill_tactical_leaf_forcing_value",
        "mcts_distill_tactical_leaf_two_ply_value",
        "mcts_distill_tactical_leaf_two_ply",
        "mcts_distill_tactical_leaf_max_candidates",
        "mcts_distill_tactical_leaf_max_replies",
        "mcts_distill_tactical_leaf_max_followups",
        "tactical_win_value",
        "tactical_attack_threat_value",
        "tactical_block_win_value",
        "tactical_block_threat_value",
        "anchor_samples",
        "learn_rate",
        "policy_loss_weight",
        "value_loss_weight",
        "self_play_policy_loss_weight",
        "self_play_value_loss_weight",
        "self_play_draw_value",
        "self_play_temp_cutoff",
        "self_play_late_temp",
        "conversion_replay_size",
        "conversion_replay_fraction",
        "conversion_replay_extra_steps",
        "conversion_replay_batch_size",
        "conversion_replay_epochs",
        "conversion_replay_policy_loss_weight",
        "conversion_replay_value_loss_weight",
        "conversion_teacher_replay_size",
        "conversion_teacher_value",
        "conversion_teacher_extra_steps",
        "conversion_teacher_batch_size",
        "conversion_teacher_epochs",
        "conversion_teacher_policy_loss_weight",
        "conversion_teacher_value_loss_weight",
        "conversion_teacher_add_to_replay",
        "conversion_teacher_depth",
        "conversion_teacher_reply_limit",
        "conversion_teacher_followup_limit",
        "c_puct",
        "temp",
        "policy_tactical_guard",
        "policy_dirichlet_alpha",
        "policy_noise_frac",
        "beam_width",
        "beam_policy_top_k",
        "beam_value_weight",
        "beam_policy_weight",
        "beam_tactical_weight",
        "beam_reply_penalty",
        "beam_fork_weight",
        "beam_fork_threshold",
        "seed",
        "init_from",
    ]
    return {key: config[key] for key in keys if key in config}


def _artifact_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _training_state_key(config):
    return (
        f"{config['board_width']}x{config['board_height']}"
        f"_n{config['n_in_row']}"
        f"_r{config['num_res_blocks']}"
        f"_f{config['num_filters']}"
    )


def _replay_path(checkpoint_dir, config):
    return Path(checkpoint_dir) / f"replay_{_training_state_key(config)}.pkl"


def _optimizer_path(model_path):
    return Path(model_path).with_suffix(".optimizer.pt")


def _relative_path(path):
    path = Path(path)
    if path.is_absolute():
        try:
            return path.relative_to(Path.cwd()).as_posix()
        except ValueError:
            return path.as_posix()
    return path.as_posix()


def _sample_matches_config(sample, config):
    try:
        state, mcts_prob, _winner = sample
    except (TypeError, ValueError):
        return False
    return (
        np.shape(state) == (4, config["board_width"], config["board_height"])
        and np.shape(mcts_prob) == (config["board_width"] * config["board_height"],)
    )


def _load_replay_buffer(path, config, logger):
    data_buffer = deque(maxlen=config["buffer_size"])
    path = Path(path)
    metadata = {
        "replay_path": _relative_path(path),
        "replay_loaded_samples": 0,
        "replay_loaded_games": 0,
    }
    if not path.exists():
        return data_buffer, metadata

    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError) as exc:
        logger.warning("Could not load replay buffer %s: %s", path, exc)
        return data_buffer, metadata

    if isinstance(payload, dict):
        samples = payload.get("samples", [])
        metadata["replay_loaded_games"] = int(payload.get("games_recorded", 0) or 0)
    else:
        samples = payload

    samples = [sample for sample in samples if _sample_matches_config(sample, config)]
    data_buffer.extend(samples[-config["buffer_size"]:])
    metadata["replay_loaded_samples"] = len(data_buffer)
    logger.info(
        "Loaded replay buffer %s: %s samples, %s recorded games",
        path,
        metadata["replay_loaded_samples"],
        metadata["replay_loaded_games"],
    )
    return data_buffer, metadata


def _save_replay_buffer(path, data_buffer, config, games_recorded, logger):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "config": _public_config(config),
        "games_recorded": int(games_recorded),
        "samples": list(data_buffer),
    }
    try:
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
    except OSError as exc:
        logger.warning("Could not save replay buffer %s: %s", path, exc)
        return False
    logger.info("Saved replay buffer %s: %s samples", path, len(data_buffer))
    return True


def _copy_history(history):
    return {
        "bootstrap": [dict(item) for item in history.get("bootstrap", [])],
        "self_play": [dict(item) for item in history.get("self_play", [])],
        "training": [dict(item) for item in history.get("training", [])],
        "checkpoints": [dict(item) for item in history.get("checkpoints", [])],
    }


def _save_registered_checkpoint(
    policy_value_net,
    config,
    checkpoint_dir,
    tag,
    games_trained,
    metrics,
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / _checkpoint_name(config["preset"], tag, games_trained)
    policy_value_net.save_model(str(model_path))
    metrics = dict(metrics)
    training_state = dict(metrics.get("training_state") or {})
    optimizer_path = _optimizer_path(model_path)
    if policy_value_net.save_optimizer(str(optimizer_path)):
        training_state["optimizer_path"] = _relative_path(optimizer_path)
    if training_state:
        metrics["training_state"] = training_state

    entry = register_checkpoint(
        model_path=str(model_path),
        registry_path=str(checkpoint_dir / "registry.json"),
        name=f"{config['preset']} {tag} ({games_trained} games)",
        preset=config["preset"],
        board_width=config["board_width"],
        board_height=config["board_height"],
        n_in_row=config["n_in_row"],
        num_res_blocks=config["num_res_blocks"],
        num_filters=config["num_filters"],
        architecture=config.get("architecture", "residual"),
        games_trained=games_trained,
        n_playout=config["eval_n_playout"],
        elo=metrics.get("elo", 1000),
        metrics=metrics,
    )
    return entry


def run_baseline_training(
    preset="shopping_baseline",
    checkpoint_dir="checkpoints",
    init_agent=None,
    resume_best=False,
):
    """Run a bounded local trainer that produces frontend-playable checkpoints."""
    config = get_training_preset(preset)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    registry_path = checkpoint_dir / "registry.json"

    logger = setup_logger("train.log")
    logger = logging.getLogger("baseline")
    logger.info("Starting %s baseline with config: %s", preset, config)
    append_training_event({
        "event": "run_start",
        "preset": preset,
        "config": _public_config(config),
    }, checkpoint_dir)

    seed = config.get("seed")
    if seed is None:
        seed = time.time_ns() % (2**32 - 1)
        config["seed"] = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info("Using random seed for %s: %s", preset, seed)

    init_model_path = None
    if init_agent is not None:
        init_model_path = resolve_model_path(init_agent)
    elif resume_best:
        init_agent = best_compatible_model_checkpoint(config, registry_path=str(registry_path))
        if init_agent is not None:
            init_model_path = resolve_model_path(init_agent)

    if init_agent is not None:
        config["init_from"] = {
            "id": init_agent["id"],
            "name": init_agent.get("name", init_agent["id"]),
            "path": init_agent["path"],
            "elo": init_agent.get("elo"),
            "games_trained": init_agent.get("games_trained"),
        }
        logger.info("Initializing %s from checkpoint: %s", preset, config["init_from"])
        append_training_event({
            "event": "init_from_checkpoint",
            "preset": preset,
            "checkpoint_id": init_agent["id"],
            "checkpoint_elo": init_agent.get("elo"),
            "games_trained": init_agent.get("games_trained"),
        }, checkpoint_dir)
    elif resume_best:
        logger.info("No compatible checkpoint found for %s; starting from random weights.", preset)

    policy_value_net = PolicyValueNet(
        config["board_width"],
        config["board_height"],
        model_file=None if init_model_path is None else str(init_model_path),
        use_gpu=config["use_gpu"],
        num_res_blocks=config["num_res_blocks"],
        num_filters=config["num_filters"],
        architecture=config.get("architecture", "residual"),
    )
    optimizer_loaded_from = None
    if init_agent is not None:
        optimizer_state_path = (
            init_agent.get("metrics", {})
            .get("training_state", {})
            .get("optimizer_path")
        )
        if optimizer_state_path:
            optimizer_state_path = _artifact_path(optimizer_state_path)
            if optimizer_state_path.exists():
                try:
                    policy_value_net.load_optimizer(str(optimizer_state_path), config["learn_rate"])
                    optimizer_loaded_from = _relative_path(optimizer_state_path)
                    logger.info("Loaded optimizer state from %s", optimizer_loaded_from)
                except (OSError, RuntimeError, ValueError) as exc:
                    logger.warning("Could not load optimizer state %s: %s", optimizer_state_path, exc)

    base_games_trained = int(config.get("init_from", {}).get("games_trained") or 0)
    replay_file = _replay_path(checkpoint_dir, config)
    data_buffer, replay_metadata = _load_replay_buffer(replay_file, config, logger)
    conversion_replay_buffer = deque(maxlen=int(config.get("conversion_replay_size", 0) or 0))
    conversion_teacher_buffer = deque(maxlen=int(config.get("conversion_teacher_replay_size", 0) or 0))
    replay_games_recorded = replay_metadata["replay_loaded_games"]
    history = {"bootstrap": [], "self_play": [], "training": [], "checkpoints": []}
    last_train_metrics = {}

    anchor_samples = int(config.get("anchor_samples", 0) or 0)
    if init_model_path is not None and anchor_samples and data_buffer:
        anchor_policy = PolicyValueNet(
            config["board_width"],
            config["board_height"],
            model_file=str(init_model_path),
            use_gpu=False,
            num_res_blocks=config["num_res_blocks"],
            num_filters=config["num_filters"],
            architecture=checkpoint_architecture(init_agent),
        )
        anchor_data = _anchor_distillation_data(
            anchor_policy,
            data_buffer,
            config,
            seed + 20_000,
            anchor_samples,
        )
        data_buffer.extend(anchor_data)
        if anchor_data:
            anchor_metrics = _train_step_from_buffer(policy_value_net, anchor_data, config)
            last_train_metrics = anchor_metrics
            anchor_record = {
                "source": "anchor_distillation",
                "game": 0,
                "total_games": base_games_trained,
                "positions": len(anchor_data),
                "buffer_size": int(len(data_buffer)),
                **_train_metric_fields(anchor_metrics),
            }
            history["bootstrap"].append({
                "source": "anchor_distillation",
                "positions": len(anchor_data),
                "buffer_size": int(len(data_buffer)),
            })
            history["training"].append(anchor_record)
            logger.info("Anchor distillation metrics: %s", anchor_record)
            append_training_event({
                "event": "train_step",
                "preset": preset,
                **anchor_record,
            }, checkpoint_dir)

    def current_training_state():
        state = {
            "replay_path": _relative_path(replay_file),
            "replay_samples": len(data_buffer),
            "replay_loaded_samples": replay_metadata["replay_loaded_samples"],
            "replay_games": replay_games_recorded,
            "replay_loaded_games": replay_metadata["replay_loaded_games"],
            "conversion_replay_samples": len(conversion_replay_buffer),
            "conversion_teacher_samples": len(conversion_teacher_buffer),
        }
        if optimizer_loaded_from:
            state["optimizer_loaded_from"] = optimizer_loaded_from
        return state

    initial_entry = _save_registered_checkpoint(
        policy_value_net,
        config,
        checkpoint_dir,
        tag="initial",
        games_trained=base_games_trained,
        metrics={
            "elo": 1000,
            "config": _public_config(config),
            "history": _copy_history(history),
            "init_from": config.get("init_from"),
            "training_state": current_training_state(),
            "last_train": last_train_metrics,
            "note": "Initial checkpoint for this run; may be random or loaded from init_from.",
        },
    )
    logger.info("Saved initial checkpoint: %s", initial_entry["path"])
    append_training_event({
        "event": "checkpoint_saved",
        "preset": preset,
        "tag": "initial",
        "checkpoint_id": initial_entry["id"],
        "elo": initial_entry["elo"],
        "games_trained": initial_entry["games_trained"],
    }, checkpoint_dir)

    saved_entries = [initial_entry]
    history["checkpoints"].append({
        "tag": "initial",
        "game": 0,
        "total_games": base_games_trained,
        "path": initial_entry["path"],
        "elo": initial_entry["elo"],
    })

    expert_games = int(config.get("expert_games", 0) or 0)
    expert_buffer = deque(maxlen=config["buffer_size"])
    for expert_idx in range(1, expert_games + 1):
        start = time.time()
        winner, expert_data, moves = _heuristic_play_data(config, seed + expert_idx)
        duration_s = time.time() - start
        expert_data = list(expert_data)
        augmented_expert_data = augment_play_data(
            expert_data,
            config["board_height"],
            config["board_width"],
        )
        data_buffer.extend(augmented_expert_data)
        expert_buffer.extend(augmented_expert_data)
        replay_games_recorded += 1
        bootstrap_record = {
            "source": "heuristic",
            "game": expert_idx,
            "winner": int(winner),
            "moves": int(len(moves)),
            "buffer_size": int(len(data_buffer)),
            "duration_s": round(duration_s, 3),
        }
        history["bootstrap"].append(bootstrap_record)
        append_training_event({
            "event": "bootstrap_game",
            "preset": preset,
            **bootstrap_record,
        }, checkpoint_dir)
        logger.info(
            "Heuristic bootstrap game %s/%s: winner=%s moves=%s buffer=%s duration=%.2fs",
            expert_idx,
            expert_games,
            winner,
            len(moves),
            len(data_buffer),
            duration_s,
        )

    if expert_buffer:
        last_train_metrics = _train_step_from_buffer(policy_value_net, expert_buffer, config)
        bootstrap_train_record = {
            "source": "heuristic_bootstrap",
            "game": 0,
            "total_games": base_games_trained,
            **_train_metric_fields(last_train_metrics),
        }
        history["training"].append(bootstrap_train_record)
        append_training_event({
            "event": "train_step",
            "preset": preset,
            **bootstrap_train_record,
        }, checkpoint_dir)
        logger.info("Heuristic bootstrap training metrics: %s", last_train_metrics)

    tactical_puzzles = int(config.get("tactical_puzzles", 0) or 0)
    if tactical_puzzles:
        start = time.time()
        tactical_data = _tactical_puzzle_data(config, seed + 10_000, tactical_puzzles)
        data_buffer.extend(tactical_data)
        replay_games_recorded += tactical_puzzles
        tactical_record = {
            "source": "tactical_puzzle",
            "positions": tactical_puzzles,
            "buffer_size": int(len(data_buffer)),
            "duration_s": round(time.time() - start, 3),
        }
        history["bootstrap"].append(tactical_record)
        last_train_metrics = _train_step_from_buffer(policy_value_net, tactical_data, config)
        tactical_train_record = {
            "source": "tactical_puzzle",
            "game": 0,
            "total_games": base_games_trained,
            **_train_metric_fields(last_train_metrics),
        }
        history["training"].append(tactical_train_record)
        append_training_event({
            "event": "tactical_puzzles",
            "preset": preset,
            **tactical_record,
        }, checkpoint_dir)
        append_training_event({
            "event": "train_step",
            "preset": preset,
            **tactical_train_record,
        }, checkpoint_dir)
        logger.info(
            "Tactical puzzle bootstrap: positions=%s buffer=%s metrics=%s",
            tactical_puzzles,
            len(data_buffer),
            last_train_metrics,
        )

    hard_position_puzzles = int(config.get("hard_position_puzzles", 0) or 0)
    if hard_position_puzzles:
        start = time.time()
        hard_position_data = _hard_position_puzzle_data(
            config,
            seed + 15_000,
            hard_position_puzzles,
        )
        data_buffer.extend(hard_position_data)
        replay_games_recorded += hard_position_puzzles
        hard_position_record = {
            "source": "hard_position_puzzle",
            "positions": hard_position_puzzles,
            "buffer_size": int(len(data_buffer)),
            "duration_s": round(time.time() - start, 3),
        }
        history["bootstrap"].append(hard_position_record)
        last_train_metrics = _train_step_from_buffer(policy_value_net, hard_position_data, config)
        hard_position_train_record = {
            "source": "hard_position_puzzle",
            "game": 0,
            "total_games": base_games_trained,
            **_train_metric_fields(last_train_metrics),
        }
        history["training"].append(hard_position_train_record)
        append_training_event({
            "event": "hard_position_puzzles",
            "preset": preset,
            **hard_position_record,
        }, checkpoint_dir)
        append_training_event({
            "event": "train_step",
            "preset": preset,
            **hard_position_train_record,
        }, checkpoint_dir)
        logger.info(
            "Hard-position puzzle bootstrap: positions=%s buffer=%s metrics=%s",
            hard_position_puzzles,
            len(data_buffer),
            last_train_metrics,
        )

    threat_space_puzzles = int(config.get("threat_space_puzzles", 0) or 0)
    if threat_space_puzzles:
        start = time.time()
        threat_space_data = _threat_space_puzzle_data(
            config,
            seed + 17_000,
            threat_space_puzzles,
        )
        data_buffer.extend(threat_space_data)
        replay_games_recorded += threat_space_puzzles
        threat_space_record = {
            "source": "threat_space_puzzle",
            "positions": threat_space_puzzles,
            "buffer_size": int(len(data_buffer)),
            "duration_s": round(time.time() - start, 3),
        }
        history["bootstrap"].append(threat_space_record)
        last_train_metrics = _train_step_from_buffer(policy_value_net, threat_space_data, config)
        threat_space_train_record = {
            "source": "threat_space_puzzle",
            "game": 0,
            "total_games": base_games_trained,
            **_train_metric_fields(last_train_metrics),
        }
        history["training"].append(threat_space_train_record)
        append_training_event({
            "event": "threat_space_puzzles",
            "preset": preset,
            **threat_space_record,
        }, checkpoint_dir)
        append_training_event({
            "event": "train_step",
            "preset": preset,
            **threat_space_train_record,
        }, checkpoint_dir)
        logger.info(
            "Threat-space puzzle bootstrap: positions=%s buffer=%s metrics=%s",
            threat_space_puzzles,
            len(data_buffer),
            last_train_metrics,
        )

    threat_space_proof_positions = int(config.get("threat_space_proof_positions", 0) or 0)
    if threat_space_proof_positions:
        start = time.time()
        proof_data, proof_stats = _threat_space_proof_value_data(
            config,
            seed + 17_500,
            threat_space_proof_positions,
        )
        if config.get("threat_space_proof_add_to_replay", False):
            data_buffer.extend(proof_data)
            replay_games_recorded += threat_space_proof_positions
        proof_record = {
            "source": "threat_space_proof",
            "positions": threat_space_proof_positions,
            "buffer_size": int(len(data_buffer)),
            "duration_s": round(time.time() - start, 3),
            **proof_stats,
        }
        history["bootstrap"].append(proof_record)
        append_training_event({
            "event": "threat_space_proof_values",
            "preset": preset,
            **proof_record,
        }, checkpoint_dir)
        if proof_data:
            proof_config = _threat_space_proof_train_config(config)
            last_train_metrics = _train_step_from_buffer(
                policy_value_net,
                proof_data,
                proof_config,
                priority_fraction=0.0,
                policy_loss_weight=config.get("threat_space_proof_policy_loss_weight", 0.0),
                value_loss_weight=config.get("threat_space_proof_value_loss_weight", 1.0),
            )
            proof_train_record = {
                "source": "threat_space_proof",
                "game": 0,
                "total_games": base_games_trained,
                **proof_stats,
                **_train_metric_fields(last_train_metrics),
            }
            history["training"].append(proof_train_record)
            append_training_event({
                "event": "train_step",
                "preset": preset,
                **proof_train_record,
            }, checkpoint_dir)
        else:
            logger.warning(
                "Threat-space proof value bootstrap generated no samples from %s positions",
                threat_space_proof_positions,
            )
        logger.info(
            "Threat-space proof value bootstrap: positions=%s samples=%s buffer=%s stats=%s metrics=%s",
            threat_space_proof_positions,
            len(proof_data),
            len(data_buffer),
            proof_stats,
            last_train_metrics,
        )

    mcts_distill_positions = int(config.get("mcts_distill_positions", 0) or 0)
    if mcts_distill_positions:
        start = time.time()
        mcts_distill_data, mcts_distill_stats = _mcts_distill_data(
            policy_value_net,
            config,
            seed + 18_000,
            mcts_distill_positions,
        )
        data_buffer.extend(mcts_distill_data)
        replay_games_recorded += mcts_distill_positions
        mcts_distill_record = {
            "source": "mcts_distill",
            "positions": len(mcts_distill_data),
            "buffer_size": int(len(data_buffer)),
            "duration_s": round(time.time() - start, 3),
            **mcts_distill_stats,
        }
        history["bootstrap"].append(mcts_distill_record)
        append_training_event({
            "event": "mcts_distill_positions",
            "preset": preset,
            **mcts_distill_record,
        }, checkpoint_dir)
        if mcts_distill_data:
            distill_config = _mcts_distill_train_config(config)
            last_train_metrics = _train_step_from_buffer(
                policy_value_net,
                mcts_distill_data,
                distill_config,
                priority_fraction=0.0,
                policy_loss_weight=config.get("mcts_distill_policy_loss_weight", 1.0),
                value_loss_weight=config.get("mcts_distill_value_loss_weight", 0.25),
            )
            mcts_distill_train_record = {
                "source": "mcts_distill",
                "game": 0,
                "total_games": base_games_trained,
                "mcts_distill_samples": len(mcts_distill_data),
                **mcts_distill_stats,
                **_train_metric_fields(last_train_metrics),
            }
            history["training"].append(mcts_distill_train_record)
            append_training_event({
                "event": "train_step",
                "preset": preset,
                **mcts_distill_train_record,
            }, checkpoint_dir)
        else:
            logger.warning(
                "MCTS distillation accepted no positions after %s attempts; "
                "skipping distillation train step",
                mcts_distill_stats.get("mcts_distill_attempts", 0),
            )
        logger.info(
            "MCTS distillation bootstrap: positions=%s buffer=%s stats=%s metrics=%s",
            len(mcts_distill_data),
            len(data_buffer),
            mcts_distill_stats,
            last_train_metrics,
        )

    for game_idx in range(1, config["self_play_games"] + 1):
        board = _new_board(config)
        game = Game(board)
        self_play_mode = config.get("self_play_mode", "mcts")
        self_play_player = _new_self_play_player(
            policy_value_net,
            config,
            seed + 30_000 + game_idx,
        )
        start = time.time()
        winner, play_data, moves = game.start_self_play(
            self_play_player,
            is_shown=0,
            temp=_self_play_temperature(config),
        )
        duration_s = time.time() - start
        play_data = list(play_data)
        play_data = _apply_self_play_draw_value(
            play_data,
            winner,
            config.get("self_play_draw_value", 0.0),
        )
        augmented_self_play_data = augment_play_data(
            play_data,
            config["board_height"],
            config["board_width"],
        )
        conversion_teacher_data = _conversion_teacher_data(
            config,
            moves,
            value=config.get("conversion_teacher_value", 0.95),
        )
        augmented_conversion_teacher_data = augment_play_data(
            conversion_teacher_data,
            config["board_height"],
            config["board_width"],
        ) if conversion_teacher_data else []
        data_buffer.extend(augmented_self_play_data)
        if winner != -1 and conversion_replay_buffer.maxlen:
            conversion_replay_buffer.extend(augmented_self_play_data)
        if augmented_conversion_teacher_data and conversion_teacher_buffer.maxlen:
            conversion_teacher_buffer.extend(augmented_conversion_teacher_data)
            if config.get("conversion_teacher_add_to_replay", True):
                data_buffer.extend(augmented_conversion_teacher_data)
        replay_games_recorded += 1

        main_train_metrics = {}
        conversion_train_metrics = {}
        conversion_teacher_metrics = {}
        if len(data_buffer) >= config["batch_size"]:
            main_train_metrics = _train_step_from_buffer(
                policy_value_net,
                data_buffer,
                config,
                priority_buffer=conversion_replay_buffer,
                policy_loss_weight=config.get(
                    "self_play_policy_loss_weight",
                    config.get("policy_loss_weight", 1.0),
                ),
                value_loss_weight=config.get(
                    "self_play_value_loss_weight",
                    config.get("value_loss_weight", 1.0),
                ),
            )
            last_train_metrics = main_train_metrics
            train_record = {
                "game": game_idx,
                "total_games": base_games_trained + game_idx,
                "conversion_replay_samples": len(conversion_replay_buffer),
                **_train_metric_fields(main_train_metrics),
            }
            history["training"].append(train_record)
            append_training_event({
                "event": "train_step",
                "preset": preset,
                **train_record,
                }, checkpoint_dir)

        conversion_extra_steps = int(config.get("conversion_replay_extra_steps", 0) or 0)
        if conversion_extra_steps and conversion_replay_buffer:
            conversion_config = _conversion_replay_train_config(config)
            for step_idx in range(1, conversion_extra_steps + 1):
                conversion_train_metrics = _train_step_from_buffer(
                    policy_value_net,
                    conversion_replay_buffer,
                    conversion_config,
                    priority_fraction=0.0,
                    policy_loss_weight=config.get("conversion_replay_policy_loss_weight", 1.0),
                    value_loss_weight=config.get("conversion_replay_value_loss_weight", 0.0),
                )
                last_train_metrics = conversion_train_metrics
                conversion_train_record = {
                    "source": "conversion_replay",
                    "game": game_idx,
                    "step": step_idx,
                    "total_games": base_games_trained + game_idx,
                    "conversion_replay_samples": len(conversion_replay_buffer),
                    **_train_metric_fields(conversion_train_metrics),
                }
                history["training"].append(conversion_train_record)
                append_training_event({
                    "event": "train_step",
                    "preset": preset,
                    **conversion_train_record,
                }, checkpoint_dir)

        conversion_teacher_extra_steps = int(config.get("conversion_teacher_extra_steps", 0) or 0)
        if conversion_teacher_extra_steps and conversion_teacher_buffer:
            teacher_config = _conversion_teacher_train_config(config)
            for step_idx in range(1, conversion_teacher_extra_steps + 1):
                conversion_teacher_metrics = _train_step_from_buffer(
                    policy_value_net,
                    conversion_teacher_buffer,
                    teacher_config,
                    priority_fraction=0.0,
                    policy_loss_weight=config.get("conversion_teacher_policy_loss_weight", 1.0),
                    value_loss_weight=config.get("conversion_teacher_value_loss_weight", 0.5),
                )
                last_train_metrics = conversion_teacher_metrics
                conversion_teacher_record = {
                    "source": "conversion_teacher",
                    "game": game_idx,
                    "step": step_idx,
                    "total_games": base_games_trained + game_idx,
                    "conversion_teacher_positions": len(conversion_teacher_data),
                    "conversion_teacher_samples": len(conversion_teacher_buffer),
                    **_train_metric_fields(conversion_teacher_metrics),
                }
                history["training"].append(conversion_teacher_record)
                append_training_event({
                    "event": "train_step",
                    "preset": preset,
                    **conversion_teacher_record,
                }, checkpoint_dir)

        self_play_record = {
            "game": game_idx,
            "total_games": base_games_trained + game_idx,
            "winner": int(winner),
            "moves": int(len(moves)),
            "buffer_size": int(len(data_buffer)),
            "conversion_replay_samples": len(conversion_replay_buffer),
            "conversion_teacher_positions": len(conversion_teacher_data),
            "conversion_teacher_samples": len(conversion_teacher_buffer),
            "duration_s": round(duration_s, 3),
        }
        self_play_record.update(_self_play_stats(self_play_player, self_play_mode))
        if main_train_metrics:
            self_play_record.update(_train_metric_fields(main_train_metrics))
        if conversion_train_metrics:
            self_play_record["conversion_replay_train_steps"] = conversion_extra_steps
            self_play_record["conversion_replay_policy_loss"] = float(
                conversion_train_metrics.get("policy_loss", 0.0)
            )
            self_play_record["conversion_replay_value_loss"] = float(
                conversion_train_metrics.get("value_loss", 0.0)
            )
        if conversion_teacher_metrics:
            self_play_record["conversion_teacher_train_steps"] = conversion_teacher_extra_steps
            self_play_record["conversion_teacher_policy_loss"] = float(
                conversion_teacher_metrics.get("policy_loss", 0.0)
            )
            self_play_record["conversion_teacher_value_loss"] = float(
                conversion_teacher_metrics.get("value_loss", 0.0)
            )
        history["self_play"].append(self_play_record)
        append_training_event({
            "event": "self_play_game",
            "preset": preset,
            **self_play_record,
        }, checkpoint_dir)

        logger.info(
            "Self-play game %s/%s: mode=%s winner=%s moves=%s buffer=%s duration=%.2fs stats=%s main_metrics=%s conversion_metrics=%s teacher_metrics=%s",
            game_idx,
            config["self_play_games"],
            self_play_mode,
            winner,
            len(moves),
            len(data_buffer),
            duration_s,
            _self_play_stats(self_play_player, self_play_mode),
            main_train_metrics,
            conversion_train_metrics,
            conversion_teacher_metrics,
        )

        if game_idx % config["save_freq"] == 0 and game_idx != config["self_play_games"]:
            replay_saved = _save_replay_buffer(
                replay_file,
                data_buffer,
                config,
                replay_games_recorded,
                logger,
            )
            training_state = current_training_state()
            training_state["replay_saved"] = replay_saved
            entry = _save_registered_checkpoint(
                policy_value_net,
                config,
                checkpoint_dir,
                tag="mid",
                games_trained=base_games_trained + game_idx,
                metrics={
                    "elo": 1000,
                    "config": _public_config(config),
                    "history": _copy_history(history),
                    "init_from": config.get("init_from"),
                    "training_state": training_state,
                    "last_train": last_train_metrics,
                    "note": "Mid-run checkpoint before final evaluation.",
                },
            )
            saved_entries.append(entry)
            history["checkpoints"].append({
                "tag": "mid",
                "game": game_idx,
                "total_games": base_games_trained + game_idx,
                "path": entry["path"],
                "elo": entry["elo"],
            })
            logger.info("Saved mid checkpoint: %s", entry["path"])
            append_training_event({
                "event": "checkpoint_saved",
                "preset": preset,
                "tag": "mid",
                "checkpoint_id": entry["id"],
                "elo": entry["elo"],
                "games_trained": entry["games_trained"],
            }, checkpoint_dir)

    replay_saved = _save_replay_buffer(
        replay_file,
        data_buffer,
        config,
        replay_games_recorded,
        logger,
    )
    training_state = current_training_state()
    training_state["replay_saved"] = replay_saved
    eval_results, elo = _evaluate_policy(policy_value_net, config)
    final_entry = _save_registered_checkpoint(
        policy_value_net,
        config,
        checkpoint_dir,
        tag="final",
        games_trained=base_games_trained + config["self_play_games"],
        metrics={
            "elo": elo,
            "config": _public_config(config),
            "eval": eval_results,
            "history": _copy_history(history),
            "init_from": config.get("init_from"),
            "training_state": training_state,
            "last_train": last_train_metrics,
        },
    )
    saved_entries.append(final_entry)
    history["checkpoints"].append({
        "tag": "final",
        "game": config["self_play_games"],
        "total_games": base_games_trained + config["self_play_games"],
        "path": final_entry["path"],
        "elo": final_entry["elo"],
    })
    logger.info("Saved final checkpoint: %s", final_entry["path"])
    logger.info("Final Elo estimate: %s; eval=%s", elo, eval_results)
    append_training_event({
        "event": "checkpoint_saved",
        "preset": preset,
        "tag": "final",
        "checkpoint_id": final_entry["id"],
        "elo": final_entry["elo"],
        "games_trained": final_entry["games_trained"],
        "eval": eval_results,
    }, checkpoint_dir)
    return {
        "config": config,
        "registry_path": str(checkpoint_dir / "registry.json"),
        "checkpoints": saved_entries,
        "final": final_entry,
    }


if __name__ == '__main__':
    # Set start method to spawn for CUDA compatibility if needed, though fork is default on Linux
    # mp.set_start_method('spawn') 
    training_pipeline = TrainPipeline()
    training_pipeline.run()
