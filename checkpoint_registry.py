import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_REGISTRY_PATH = Path("checkpoints") / "registry.json"

BASELINES = [
    {
        "id": "random",
        "type": "baseline",
        "name": "Random legal move",
        "elo": 800,
        "board_width": 6,
        "board_height": 6,
        "n_in_row": 4,
        "description": "Chooses uniformly from legal moves.",
    },
    {
        "id": "heuristic",
        "type": "baseline",
        "name": "One-ply heuristic",
        "elo": 1000,
        "board_width": 6,
        "board_height": 6,
        "n_in_row": 4,
        "description": "Wins immediately, blocks immediate losses, otherwise favors the center.",
    },
]


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def default_registry():
    return {
        "version": 1,
        "updated_at": utc_now_iso(),
        "baselines": [dict(item) for item in BASELINES],
        "checkpoints": [],
    }


def load_registry(registry_path=DEFAULT_REGISTRY_PATH):
    registry_path = Path(registry_path)
    if not registry_path.exists():
        return default_registry()

    with registry_path.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)

    registry.setdefault("version", 1)
    registry.setdefault("baselines", [dict(item) for item in BASELINES])
    registry.setdefault("checkpoints", [])
    registry.setdefault("updated_at", utc_now_iso())
    return registry


def save_registry(registry, registry_path=DEFAULT_REGISTRY_PATH):
    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry["updated_at"] = utc_now_iso()
    tmp_path = registry_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, registry_path)


def checkpoint_id_from_path(model_path):
    return Path(model_path).stem.replace(".", "_")


def _normalized_path(model_path):
    path = Path(model_path)
    if path.is_absolute():
        try:
            return path.relative_to(Path.cwd()).as_posix()
        except ValueError:
            return path.as_posix()
    return path.as_posix()


def register_checkpoint(
    model_path,
    registry_path=DEFAULT_REGISTRY_PATH,
    name=None,
    preset="shopping_baseline",
    board_width=6,
    board_height=6,
    n_in_row=4,
    num_res_blocks=2,
    num_filters=32,
    architecture="residual",
    games_trained=0,
    n_playout=8,
    elo=1000,
    metrics=None,
):
    registry = load_registry(registry_path)
    entry_id = checkpoint_id_from_path(model_path)
    entry = {
        "id": entry_id,
        "type": "checkpoint",
        "name": name or entry_id,
        "path": _normalized_path(model_path),
        "preset": preset,
        "board_width": board_width,
        "board_height": board_height,
        "n_in_row": n_in_row,
        "num_res_blocks": num_res_blocks,
        "num_filters": num_filters,
        "architecture": architecture,
        "games_trained": games_trained,
        "n_playout": n_playout,
        "elo": int(round(elo)),
        "metrics": metrics or {},
        "created_at": utc_now_iso(),
    }

    checkpoints = [
        item for item in registry["checkpoints"]
        if item.get("id") != entry_id and item.get("path") != entry["path"]
    ]
    checkpoints.append(entry)
    checkpoints.sort(key=lambda item: (item.get("games_trained", 0), item.get("created_at", "")))
    registry["checkpoints"] = checkpoints
    save_registry(registry, registry_path)
    return entry


def update_checkpoint(entry_id, registry_path=DEFAULT_REGISTRY_PATH, updates=None, metrics_updates=None):
    registry = load_registry(registry_path)
    updated_entry = None
    for item in registry["checkpoints"]:
        if item.get("id") == entry_id:
            if updates:
                item.update(updates)
            if metrics_updates:
                item.setdefault("metrics", {}).update(metrics_updates)
            item["updated_at"] = utc_now_iso()
            updated_entry = item
            break

    if updated_entry is None:
        raise ValueError(f"Checkpoint '{entry_id}' not found")

    save_registry(registry, registry_path)
    return updated_entry


def list_agents(registry_path=DEFAULT_REGISTRY_PATH):
    registry = load_registry(registry_path)
    agents = []
    agents.extend(registry.get("baselines", []))
    agents.extend(registry.get("checkpoints", []))
    return agents


def find_agent(agent_id, registry_path=DEFAULT_REGISTRY_PATH):
    for agent in list_agents(registry_path):
        if agent.get("id") == agent_id:
            return agent
    return None


def latest_checkpoint(registry_path=DEFAULT_REGISTRY_PATH):
    checkpoints = load_registry(registry_path).get("checkpoints", [])
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: (item.get("created_at", ""), item.get("games_trained", 0)))


def compatible_checkpoints(reference, registry_path=DEFAULT_REGISTRY_PATH):
    registry = load_registry(registry_path)
    return [
        item for item in registry.get("checkpoints", [])
        if item.get("board_width") == reference.get("board_width")
        and item.get("board_height") == reference.get("board_height")
        and item.get("n_in_row") == reference.get("n_in_row")
    ]


def checkpoint_architecture(item):
    return (
        item.get("architecture")
        or item.get("metrics", {}).get("config", {}).get("architecture")
        or "residual"
    )


def _promotion(item):
    return item.get("metrics", {}).get("promotion", {}) or {}


def _promotion_checks(item):
    return _promotion(item).get("promotion_checks", {}) or {}


def has_current_promotion_evaluation(item):
    checks = _promotion_checks(item)
    return bool(checks) and all((
        checks.get("runtime"),
        checks.get("heuristic_games"),
        checks.get("previous_best_games"),
    ))


def has_current_promotion(item):
    checks = _promotion_checks(item)
    if not bool(_promotion(item).get("promoted")) or not bool(checks):
        return False

    required = [
        "runtime",
        "heuristic_games",
        "heuristic_score",
        "previous_best_games",
        "previous_best_score",
        "elo_floor",
    ]
    required.append("improvement" if "improvement" in checks else "elo_improved")
    return all(bool(checks.get(key)) for key in required)


def checkpoint_strength_key(item):
    promotion = _promotion(item)
    metrics = item.get("metrics", {}) or {}
    evaluated = bool(promotion)
    trained = bool(metrics.get("last_train") or metrics.get("history", {}).get("training"))
    promoted = has_current_promotion(item)
    promotion_time = item.get("updated_at") or item.get("created_at", "")
    return (
        bool(promotion.get("gate_passed")),
        promoted,
        item.get("elo", 0),
        promotion_time if promoted else "",
        has_current_promotion_evaluation(item),
        evaluated,
        trained,
        item.get("games_trained", 0),
        item.get("created_at", ""),
    )


def champion_checkpoint(reference, registry_path=DEFAULT_REGISTRY_PATH, exclude_id=None):
    candidates = [
        item for item in compatible_checkpoints(reference, registry_path)
        if item.get("id") != exclude_id
    ]
    if not candidates:
        return None

    return max(candidates, key=checkpoint_strength_key)


def best_checkpoint(registry_path=DEFAULT_REGISTRY_PATH, board_width=16, board_height=16, n_in_row=5):
    reference = {
        "board_width": board_width,
        "board_height": board_height,
        "n_in_row": n_in_row,
    }
    candidates = compatible_checkpoints(reference, registry_path)
    if not candidates:
        checkpoints = load_registry(registry_path).get("checkpoints", [])
        return latest_checkpoint(registry_path) if checkpoints else None

    return max(candidates, key=checkpoint_strength_key)


def best_compatible_model_checkpoint(config, registry_path=DEFAULT_REGISTRY_PATH):
    reference = {
        "board_width": config.get("board_width"),
        "board_height": config.get("board_height"),
        "n_in_row": config.get("n_in_row"),
    }
    candidates = [
        item for item in compatible_checkpoints(reference, registry_path)
        if item.get("num_res_blocks") == config.get("num_res_blocks")
        and item.get("num_filters") == config.get("num_filters")
        and checkpoint_architecture(item) == config.get("architecture", "residual")
    ]
    if not candidates:
        return None

    return max(candidates, key=checkpoint_strength_key)


def resolve_model_path(entry):
    model_path = Path(entry["path"])
    if model_path.is_absolute():
        return model_path
    return Path.cwd() / model_path


def score_from_record(wins, draws, losses):
    games = wins + draws + losses
    if games == 0:
        return 0.0
    return (wins + 0.5 * draws) / games


def elo_after_games(initial_elo, results, k_factor=32):
    """Update Elo from [(opponent_elo, score_rate, games), ...]."""
    elo = float(initial_elo)
    for opponent_elo, score_rate, games in results:
        games = max(int(games), 0)
        if games == 0:
            continue
        expected = 1.0 / (1.0 + math.pow(10.0, (float(opponent_elo) - elo) / 400.0))
        elo += k_factor * games * (float(score_rate) - expected)
    return int(round(elo))
