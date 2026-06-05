import json
import mimetypes
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from checkpoint_registry import (
    best_checkpoint,
    checkpoint_architecture,
    checkpoint_strength_key,
    find_agent,
    list_agents,
    load_registry,
    resolve_model_path,
)
from config import (
    AVERAGE_HUMAN_GATE,
    COMPLETION_GATE,
    IMPROVEMENT_PRESETS,
    LONG_IMPROVEMENT_PRESETS,
    PROMOTION_GATE,
    TOP_HUMAN_GATE,
    TRAINING_PRESETS,
)
from game import Board
from mcts import MCTSPlayer
from model import PolicyValueNet
from players import HeuristicPlayer, RandomPlayer


STATIC_DIR = Path(__file__).parent / "static"
GAMES = {}
POLICY_CACHE = {}
TRAIN_LOG_PATH = Path("train.log")
TRAIN_EVENT_LOG_PATH = Path("checkpoints") / "training_log.jsonl"
REGISTRY_PATH = Path("checkpoints") / "registry.json"
CONFIG_PATH = Path("config.py")


def _json_response(handler, payload, status=200):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _error(handler, message, status=400):
    _json_response(handler, {"error": message}, status=status)


def _default_agent_id():
    checkpoint = best_checkpoint()
    if checkpoint is not None:
        return checkpoint["id"]
    return "heuristic"


def _tail_lines(path, max_lines=80):
    path = Path(path)
    if not path.exists():
        return []
    lines = deque(maxlen=max_lines)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            lines.append(line.rstrip("\n"))
    return list(lines)


def _leaderboard():
    registry = load_registry(REGISTRY_PATH)
    checkpoints = [
        item for item in registry.get("checkpoints", [])
        if item.get("board_width") == COMPLETION_GATE["board_width"]
        and item.get("board_height") == COMPLETION_GATE["board_height"]
        and item.get("n_in_row") == COMPLETION_GATE["n_in_row"]
    ]

    checkpoints.sort(key=checkpoint_strength_key, reverse=True)
    return checkpoints


def _training_status():
    best = best_checkpoint(
        board_width=COMPLETION_GATE["board_width"],
        board_height=COMPLETION_GATE["board_height"],
        n_in_row=COMPLETION_GATE["n_in_row"],
    )
    text_log_tail = _tail_lines(TRAIN_LOG_PATH, max_lines=56)
    event_log_tail = _tail_lines(TRAIN_EVENT_LOG_PATH, max_lines=24)
    log_tail = list(text_log_tail)
    if event_log_tail:
        if log_tail:
            log_tail.append("")
        log_tail.append(f"Structured events: {TRAIN_EVENT_LOG_PATH}")
        log_tail.extend(event_log_tail)
    return {
        "goal": {
            "title": "Top-Human Wuziqi Agent",
            "status": "active",
            "completion_gate": COMPLETION_GATE,
            "average_human_milestone": AVERAGE_HUMAN_GATE,
            "promotion_gate": PROMOTION_GATE,
        },
        "best_agent_id": None if best is None else best["id"],
        "best_agent": best,
        "leaderboard": _leaderboard(),
        "log_path": str(TRAIN_LOG_PATH),
        "event_log_path": str(TRAIN_EVENT_LOG_PATH),
        "registry_path": str(REGISTRY_PATH),
        "config_path": str(CONFIG_PATH),
        "log_tail": log_tail,
    }


def _agent_config(agent_id):
    agent = find_agent(agent_id)
    if agent is None:
        raise ValueError(f"Unknown agent '{agent_id}'")
    return agent


def _load_policy(agent):
    model_path = resolve_model_path(agent)
    cache_key = (
        agent["id"],
        str(model_path),
        model_path.stat().st_mtime if model_path.exists() else None,
        checkpoint_architecture(agent),
    )
    if cache_key not in POLICY_CACHE:
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {model_path}")
        POLICY_CACHE.clear()
        POLICY_CACHE[cache_key] = PolicyValueNet(
            agent["board_width"],
            agent["board_height"],
            model_file=str(model_path),
            use_gpu=False,
            num_res_blocks=agent.get("num_res_blocks", 4),
            num_filters=agent.get("num_filters", 64),
            architecture=checkpoint_architecture(agent),
        )
    return POLICY_CACHE[cache_key]


def _agent_mcts_tactical_threshold(agent):
    return (
        agent.get("mcts_tactical_threshold")
        or agent.get("metrics", {}).get("config", {}).get("mcts_tactical_threshold")
    )


def _agent_mcts_tactical_prior(agent):
    config = agent.get("metrics", {}).get("config", {})
    return {
        "weight": config.get("mcts_tactical_prior_weight", 0.0),
        "temperature": config.get("mcts_tactical_prior_temperature", 1.0),
        "two_ply_bonus": config.get("mcts_tactical_prior_two_ply_bonus", 0.0),
        "two_ply_max_candidates": config.get("mcts_tactical_prior_two_ply_max_candidates", 16),
        "two_ply_max_replies": config.get("mcts_tactical_prior_two_ply_max_replies", 6),
        "two_ply_max_followups": config.get("mcts_tactical_prior_two_ply_max_followups", 12),
    }


def _agent_mcts_tactical_leaf(agent):
    config = agent.get("metrics", {}).get("config", {})
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


def _build_agent(agent):
    if agent["type"] == "baseline":
        if agent["id"] == "random":
            return RandomPlayer()
        return HeuristicPlayer()

    policy = _load_policy(agent)
    tactical_prior = _agent_mcts_tactical_prior(agent)
    tactical_leaf = _agent_mcts_tactical_leaf(agent)
    return MCTSPlayer(
        policy.policy_value_fn,
        c_puct=5,
        n_playout=agent.get("n_playout", 8),
        is_selfplay=0,
        use_parallel=False,
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


def _move_to_display(board, move):
    move = int(move)
    h = move // board.width
    w = move % board.width
    return {"row": int(board.height - 1 - h), "col": int(w)}


def _display_to_move(board, row, col):
    h = board.height - 1 - int(row)
    w = int(col)
    return h * board.width + w


def _serialize_game(game_id, session):
    board = session["board"]
    end, board_winner = board.game_end()
    status = session.get("status", "active")
    winner = session.get("winner")
    if status == "active" and end:
        status = "ended"
        winner = board_winner
        session["status"] = status
        session["winner"] = winner

    stones = []
    for move, player in board.states.items():
        location = _move_to_display(board, move)
        stones.append({
            "move": int(move),
            "row": location["row"],
            "col": location["col"],
            "player": int(player),
        })

    last_move = None
    if board.last_move != -1:
        last_move = _move_to_display(board, board.last_move)
        last_move["move"] = int(board.last_move)

    return {
        "game_id": game_id,
        "status": status,
        "winner": None if winner is None else int(winner),
        "width": int(board.width),
        "height": int(board.height),
        "n_in_row": int(board.n_in_row),
        "stones": stones,
        "last_move": last_move,
        "current_player": int(board.get_current_player()),
        "human_player": int(session["human_player"]),
        "agent_player": int(session["agent_player"]),
        "agent": session["agent"],
        "message": session.get("message", ""),
    }


def _maybe_agent_move(session):
    board = session["board"]
    end, _winner = board.game_end()
    if end or session.get("status") != "active":
        return
    if board.get_current_player() != session["agent_player"]:
        return

    agent = _build_agent(session["agent"])
    agent.set_player_ind(session["agent_player"])
    move = agent.get_action(board)
    if move is None:
        return
    if move not in board.availables:
        raise ValueError(f"Agent returned illegal move: {move}")
    board.do_move(move)


def _start_game(payload):
    agent_id = payload.get("agent_id") or _default_agent_id()
    human_player = int(payload.get("human_player", 1))
    if human_player not in (1, 2):
        raise ValueError("human_player must be 1 or 2")

    agent = _agent_config(agent_id)
    board = Board(
        width=agent.get("board_width", 6),
        height=agent.get("board_height", 6),
        n_in_row=agent.get("n_in_row", 4),
    )
    board.init_board(start_player=0)
    game_id = uuid.uuid4().hex
    session = {
        "board": board,
        "status": "active",
        "winner": None,
        "human_player": human_player,
        "agent_player": 2 if human_player == 1 else 1,
        "agent": agent,
        "message": "",
    }
    GAMES[game_id] = session
    _maybe_agent_move(session)
    return _serialize_game(game_id, session)


def _play_move(payload):
    game_id = payload.get("game_id")
    if game_id not in GAMES:
        raise ValueError("Unknown game_id")

    session = GAMES[game_id]
    board = session["board"]
    if session.get("status") != "active":
        raise ValueError("Game is not active")
    if board.get_current_player() != session["human_player"]:
        raise ValueError("It is not the human player's turn")

    move = _display_to_move(board, payload.get("row"), payload.get("col"))
    if move not in board.availables:
        raise ValueError("Illegal move")

    board.do_move(move)
    _maybe_agent_move(session)
    return _serialize_game(game_id, session)


def _resign(payload):
    game_id = payload.get("game_id")
    if game_id not in GAMES:
        raise ValueError("Unknown game_id")
    session = GAMES[game_id]
    session["status"] = "resigned"
    session["winner"] = session["agent_player"]
    session["message"] = "You resigned."
    return _serialize_game(game_id, session)


class WuziqiHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/checkpoints":
            agents = list_agents()
            _json_response(self, {
                "agents": agents,
                "default_agent_id": _default_agent_id(),
            })
            return

        if path == "/api/config":
            _json_response(self, {
                "training_presets": TRAINING_PRESETS,
                "improvement_presets": IMPROVEMENT_PRESETS,
                "long_improvement_presets": LONG_IMPROVEMENT_PRESETS,
                "top_human_gate": TOP_HUMAN_GATE,
                "completion_gate": COMPLETION_GATE,
                "average_human_gate": AVERAGE_HUMAN_GATE,
                "promotion_gate": PROMOTION_GATE,
            })
            return

        if path == "/api/training-status":
            _json_response(self, _training_status())
            return

        if path == "/":
            path = "/index.html"
        self._serve_static(path)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/new-game":
                _json_response(self, _start_game(payload))
            elif path == "/api/move":
                _json_response(self, _play_move(payload))
            elif path == "/api/resign":
                _json_response(self, _resign(payload))
            else:
                _error(self, "Not found", status=404)
        except Exception as exc:
            _error(self, str(exc), status=400)

    def _read_json(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length == 0:
            return {}
        raw = self.rfile.read(content_length)
        return json.loads(raw.decode("utf-8"))

    def _serve_static(self, path):
        relative = Path(path.lstrip("/"))
        file_path = (STATIC_DIR / relative).resolve()
        if STATIC_DIR.resolve() not in file_path.parents and file_path != STATIC_DIR.resolve():
            _error(self, "Not found", status=404)
            return
        if not file_path.exists() or not file_path.is_file():
            _error(self, "Not found", status=404)
            return

        body = file_path.read_bytes()
        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(host="127.0.0.1", port=8000):
    server = ThreadingHTTPServer((host, port), WuziqiHandler)
    print(f"Wuziqi web app running at http://{host}:{port}")
    print("Use Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web app.")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
