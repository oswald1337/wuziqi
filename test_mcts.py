import random

from game import Board, Game
from checkpoint_registry import (
    best_compatible_model_checkpoint,
    has_current_promotion,
    register_checkpoint,
)
from mcts import MCTS, MCTSPlayer, tactical_leaf_value
from model import PolicyValueNet
from players import HeuristicPlayer
from tactical import (
    OPEN_FOUR_SCORE,
    OPEN_THREE_SCORE,
    WIN_SCORE,
    best_bounded_two_ply_threat_move,
    best_forcing_win_move,
    best_tactical_move,
    creates_bounded_two_ply_threat,
    creates_unanswerable_threat,
    fork_threat_count,
    line_shape,
    ranked_tactical_moves,
)
from train import (
    PolicyTacticalSelfPlayPlayer,
    TacticalBeamSelfPlayPlayer,
    _apply_self_play_draw_value,
    _conversion_replay_train_config,
    _conversion_teacher_data,
    _conversion_teacher_train_config,
    _fork_threat_puzzle_board,
    _hard_position_puzzle_sample,
    _mcts_distill_data,
    _mcts_distill_policy_target,
    _mcts_distill_position_board,
    _mcts_distill_source_config,
    _mcts_distill_train_config,
    _sample_training_batch,
    _self_play_temperature,
    _tactical_puzzle_sample,
    _threat_space_proof_train_config,
    _threat_space_proof_value_data,
    _threat_space_puzzle_board,
    _threat_space_puzzle_sample,
    _train_step_from_buffer,
)


def _flat_policy(board):
    prob = 1.0 / len(board.availables)
    return [(move, prob) for move in board.availables], 0.0


def _bad_block_policy(board):
    raw = []
    for move in board.availables:
        if move == 24:
            prob = 0.30
        elif move == 3:
            prob = 0.05
        else:
            prob = 0.01
        raw.append((move, prob))
    total = sum(prob for _move, prob in raw)
    return [(move, prob / total) for move, prob in raw], 0.0


def _place(board, move, player):
    board.states[move] = player
    board.availables.remove(move)
    board.last_move = move


def test_mcts_takes_immediate_win():
    board = Board(width=5, height=5, n_in_row=4)
    board.init_board()
    _place(board, 0, 1)
    _place(board, 1, 1)
    _place(board, 2, 1)
    board.current_player = 1

    player = MCTSPlayer(_flat_policy, n_playout=1, use_parallel=False)
    move, probs = player.get_action(board, return_prob=1)

    assert move == 3
    assert probs[3] == 1.0


def test_mcts_blocks_immediate_loss():
    board = Board(width=5, height=5, n_in_row=4)
    board.init_board()
    _place(board, 5, 2)
    _place(board, 6, 2)
    _place(board, 7, 2)
    board.current_player = 1

    player = MCTSPlayer(_flat_policy, n_playout=1, use_parallel=False)
    move, probs = player.get_action(board, return_prob=1)

    assert move == 8
    assert probs[8] == 1.0


def test_mcts_tactical_threshold_can_force_open_three():
    board = Board(width=16, height=16, n_in_row=5)
    board.init_board()
    for move in (7 * 16 + 7, 7 * 16 + 8):
        _place(board, move, 1)
    board.current_player = 1

    player = MCTSPlayer(
        _flat_policy,
        n_playout=1,
        use_parallel=False,
        tactical_threshold=OPEN_THREE_SCORE,
    )
    move, probs = player.get_action(board, return_prob=1)

    assert move in (7 * 16 + 6, 7 * 16 + 9)
    assert probs[move] == 1.0
    assert player.forced_tactical_moves == 1


def test_mcts_tactical_prior_biases_root_search_without_forcing():
    board = Board(width=16, height=16, n_in_row=5)
    board.init_board()
    for move in (7 * 16 + 7, 7 * 16 + 8):
        _place(board, move, 1)
    board.current_player = 1
    tactical_targets = {7 * 16 + 6, 7 * 16 + 9}

    plain_player = MCTSPlayer(
        _flat_policy,
        n_playout=12,
        use_parallel=False,
        tactical_threshold=WIN_SCORE,
    )
    _plain_move, plain_probs = plain_player.get_action(board, temp=1.0, return_prob=1)

    prior_player = MCTSPlayer(
        _flat_policy,
        n_playout=12,
        use_parallel=False,
        tactical_threshold=WIN_SCORE,
        tactical_prior_weight=0.9,
        tactical_prior_temperature=0.5,
    )
    move, prior_probs = prior_player.get_action(board, temp=1.0, return_prob=1)

    plain_target_mass = sum(float(plain_probs[move]) for move in tactical_targets)
    prior_target_mass = sum(float(prior_probs[move]) for move in tactical_targets)
    assert move in tactical_targets
    assert prior_target_mass > plain_target_mass
    assert prior_player.forced_tactical_moves == 0
    assert prior_player.tactical_prior_searches == 1


def test_mcts_tactical_prior_can_bias_two_ply_root_threat():
    board = Board(width=7, height=7, n_in_row=5)
    board.init_board()
    for move in (15, 17, 9, 23):
        _place(board, move, 1)
    board.current_player = 1

    player = MCTSPlayer(
        _flat_policy,
        n_playout=16,
        use_parallel=False,
        tactical_threshold=WIN_SCORE,
        tactical_prior_weight=0.95,
        tactical_prior_temperature=0.5,
        tactical_prior_two_ply_bonus=WIN_SCORE,
        tactical_prior_two_ply_max_candidates=16,
        tactical_prior_two_ply_max_replies=8,
        tactical_prior_two_ply_max_followups=16,
    )
    move, probs = player.get_action(board, temp=1.0, return_prob=1)

    assert move == 16
    assert probs[16] > 0.5
    assert player.forced_tactical_moves == 0
    assert player.tactical_prior_searches == 1
    assert player.tactical_prior_two_ply_applications == 1


def test_tactical_leaf_value_detects_forcing_values():
    board = Board(width=5, height=5, n_in_row=4)
    board.init_board()
    for move in (0, 1, 2):
        _place(board, move, 1)
    board.current_player = 1

    value, reason = tactical_leaf_value(board, return_reason=True)

    assert value == 1.0
    assert reason == "win"

    loss_board = Board(width=5, height=5, n_in_row=4)
    loss_board.init_board()
    for move in (0, 1, 2, 5, 10, 15):
        _place(loss_board, move, 1)
    loss_board.current_player = 2

    value, reason = tactical_leaf_value(loss_board, return_reason=True)

    assert value < 0
    assert reason == "multiple_immediate_losses"


def test_tactical_leaf_value_can_detect_two_ply_threats():
    board = Board(width=7, height=7, n_in_row=5)
    board.init_board()
    for move in (15, 17, 9, 23):
        _place(board, move, 1)
    board.current_player = 1

    value, reason = tactical_leaf_value(
        board,
        two_ply=True,
        two_ply_value=0.55,
        max_candidates=16,
        max_replies=8,
        max_followups=16,
        return_reason=True,
    )

    assert value == 0.55
    assert reason == "two_ply_threat"


def test_mcts_counts_tactical_leaf_reasons():
    board = Board(width=7, height=7, n_in_row=5)
    board.init_board()
    for move in (15, 17, 9, 23):
        _place(board, move, 1)
    board.current_player = 1
    mcts = MCTS(
        _flat_policy,
        n_playout=1,
        tactical_leaf_eval=True,
        tactical_leaf_two_ply=True,
        tactical_leaf_two_ply_value=0.55,
        tactical_leaf_max_candidates=16,
        tactical_leaf_max_replies=8,
        tactical_leaf_max_followups=16,
    )

    value, reason = mcts._tactical_leaf_value(board)

    assert value == 0.55
    assert reason == "two_ply_threat"
    assert mcts.tactical_leaf_reasons["two_ply_threat"] == 1


def test_mcts_tactical_leaf_value_can_override_bad_policy_prior():
    board = Board(width=5, height=5, n_in_row=4)
    board.init_board()
    for move in (0, 1, 2):
        _place(board, move, 1)
    board.current_player = 2

    plain_mcts = MCTS(
        _bad_block_policy,
        c_puct=5,
        n_playout=96,
        tactical_leaf_eval=False,
    )
    plain_acts, plain_probs = plain_mcts.get_move_probs(board, temp=1e-6)
    plain_move = plain_acts[int(max(range(len(plain_probs)), key=lambda idx: plain_probs[idx]))]

    leaf_mcts = MCTS(
        _bad_block_policy,
        c_puct=5,
        n_playout=96,
        tactical_leaf_eval=True,
    )
    leaf_acts, leaf_probs = leaf_mcts.get_move_probs(board, temp=1e-6)
    leaf_move = leaf_acts[int(max(range(len(leaf_probs)), key=lambda idx: leaf_probs[idx]))]

    assert plain_move == 24
    assert leaf_move == 3
    assert leaf_mcts.tactical_leaf_evaluations > 0
    assert leaf_mcts.tactical_leaf_positive > 0


def test_tactical_ranker_prioritizes_open_four():
    board = Board(width=16, height=16, n_in_row=5)
    board.init_board()
    for move in (5 * 16 + 5, 5 * 16 + 6, 5 * 16 + 7):
        _place(board, move, 1)
    board.current_player = 1

    ranked = ranked_tactical_moves(board)
    top_move = ranked[0]["move"]

    assert top_move in (5 * 16 + 4, 5 * 16 + 8)
    assert best_tactical_move(board) in (5 * 16 + 4, 5 * 16 + 8)


def test_heuristic_player_uses_tactical_ranker():
    board = Board(width=16, height=16, n_in_row=5)
    board.init_board()
    for move in (7 * 16 + 7, 7 * 16 + 8, 7 * 16 + 9):
        _place(board, move, 1)
    board.current_player = 1

    player = HeuristicPlayer(seed=1)
    move = player.get_action(board)

    assert move in (7 * 16 + 6, 7 * 16 + 10)


def test_policy_value_fn_blends_shape_prior():
    board = Board(width=5, height=5, n_in_row=4)
    board.init_board()
    _place(board, 12, 1)
    board.current_player = 2

    policy_value_net = PolicyValueNet(
        5,
        5,
        use_gpu=False,
        num_res_blocks=1,
        num_filters=16,
    )
    probs, _value = policy_value_net.policy_value_fn(board)
    prob_map = dict(probs)

    assert prob_map[13] > prob_map[0]


def test_conv_attention_policy_value_net_smoke():
    board = Board(width=5, height=5, n_in_row=4)
    board.init_board()
    policy_value_net = PolicyValueNet(
        5,
        5,
        use_gpu=False,
        num_res_blocks=1,
        num_filters=8,
        architecture="conv_attention",
    )

    probs, value = policy_value_net.policy_value_fn(board)

    assert len(list(probs)) == 25
    assert -1.0 <= value <= 1.0


def test_policy_value_train_step_records_loss_components():
    board = Board(width=5, height=5, n_in_row=4)
    board.init_board()
    state = board.current_state()
    target = [0.0] * 25
    target[12] = 1.0
    policy_value_net = PolicyValueNet(
        5,
        5,
        use_gpu=False,
        num_res_blocks=1,
        num_filters=8,
        architecture="conv_attention",
    )

    loss, entropy = policy_value_net.train_step(
        [state, state],
        [target, target],
        [0.0, 0.0],
        1e-3,
        policy_loss_weight=0.75,
        value_loss_weight=0.25,
    )

    components = policy_value_net.last_train_components
    assert loss > 0
    assert entropy > 0
    assert components["policy_loss"] > 0
    assert components["value_loss"] >= 0
    assert components["policy_loss_weight"] == 0.75
    assert components["value_loss_weight"] == 0.25


def test_train_step_from_buffer_allows_value_weight_override():
    board = Board(width=5, height=5, n_in_row=4)
    board.init_board()
    target = [0.0] * 25
    target[12] = 1.0
    data = [
        (board.current_state(), target, 1.0),
        (board.current_state(), target, -1.0),
    ]
    config = {
        "batch_size": 2,
        "epochs": 1,
        "learn_rate": 1e-3,
        "policy_loss_weight": 1.0,
        "value_loss_weight": 1.0,
    }
    policy_value_net = PolicyValueNet(
        5,
        5,
        use_gpu=False,
        num_res_blocks=1,
        num_filters=8,
        architecture="conv_attention",
    )

    metrics = _train_step_from_buffer(
        policy_value_net,
        data,
        config,
        value_loss_weight=0.0,
    )

    assert metrics["policy_loss_weight"] == 1.0
    assert metrics["value_loss_weight"] == 0.0
    assert metrics["policy_loss"] > 0
    assert metrics["value_loss"] >= 0


def test_sample_training_batch_mixes_priority_replay():
    base = [(f"state-{idx}", f"policy-{idx}", 0.0) for idx in range(20)]
    priority = [(f"priority-{idx}", f"policy-p{idx}", 1.0) for idx in range(10)]

    batch, priority_samples = _sample_training_batch(
        base,
        batch_size=8,
        priority_buffer=priority,
        priority_fraction=0.5,
    )

    assert len(batch) == 8
    assert priority_samples == 4
    assert sum(1 for item in batch if str(item[0]).startswith("priority-")) == 4


def test_conversion_replay_train_config_uses_small_policy_batch():
    config = {
        "batch_size": 256,
        "epochs": 3,
        "conversion_replay_batch_size": 64,
        "conversion_replay_epochs": 1,
    }

    conversion_config = _conversion_replay_train_config(config)

    assert conversion_config["batch_size"] == 64
    assert conversion_config["epochs"] == 1
    assert config["batch_size"] == 256
    assert config["epochs"] == 3


def test_conversion_teacher_train_config_uses_small_teacher_batch():
    config = {
        "batch_size": 256,
        "epochs": 3,
        "conversion_teacher_batch_size": 32,
        "conversion_teacher_epochs": 2,
    }

    teacher_config = _conversion_teacher_train_config(config)

    assert teacher_config["batch_size"] == 32
    assert teacher_config["epochs"] == 2
    assert config["batch_size"] == 256
    assert config["epochs"] == 3


def test_mcts_distill_train_config_uses_distill_batch_and_epochs():
    config = {
        "batch_size": 256,
        "epochs": 3,
        "mcts_distill_batch_size": 32,
        "mcts_distill_epochs": 2,
    }

    distill_config = _mcts_distill_train_config(config)

    assert distill_config["batch_size"] == 32
    assert distill_config["epochs"] == 2
    assert config["batch_size"] == 256
    assert config["epochs"] == 3


def test_threat_space_proof_train_config_uses_value_batch_and_epochs():
    config = {
        "batch_size": 256,
        "epochs": 3,
        "threat_space_proof_batch_size": 64,
        "threat_space_proof_epochs": 2,
    }

    proof_config = _threat_space_proof_train_config(config)

    assert proof_config["batch_size"] == 64
    assert proof_config["epochs"] == 2
    assert config["batch_size"] == 256
    assert config["epochs"] == 3


def test_mcts_distill_position_board_generates_target():
    config = {
        "board_width": 16,
        "board_height": 16,
        "n_in_row": 5,
        "mcts_distill_source": "hard_position",
        "hard_position_noise_stones": 0,
        "hard_position_fork_value": 0.91,
    }
    rng = random.Random(7)

    board, target_move, value, source = _mcts_distill_position_board(config, rng, 0)

    assert source == "hard_position"
    assert target_move in board.availables
    assert value == 0.91


def test_mcts_distill_policy_target_blends_search_and_teacher_target():
    board = Board(width=5, height=5, n_in_row=4)
    board.init_board()
    for move in (0, 1, 2):
        _place(board, move, 1)
    board.current_player = 2
    config = {
        "c_puct": 5,
        "mcts_distill_playouts": 8,
        "mcts_distill_temp": 1.0,
        "mcts_distill_target_blend": 0.5,
    }

    policy, stats = _mcts_distill_policy_target(
        type("Policy", (), {"policy_value_fn": staticmethod(_flat_policy)})(),
        board,
        3,
        config,
    )

    assert policy.shape == (25,)
    assert abs(policy.sum() - 1.0) < 1e-9
    assert policy[3] >= 0.5
    assert stats["target_mass"] >= 0.5


def test_mcts_distill_policy_target_can_emit_sharp_teacher_target():
    board = Board(width=5, height=5, n_in_row=4)
    board.init_board()
    for move in (0, 1, 2):
        _place(board, move, 1)
    board.current_player = 1
    config = {
        "c_puct": 5,
        "mcts_distill_playouts": 16,
        "mcts_distill_temp": 0.75,
        "mcts_distill_target_mode": "teacher",
        "mcts_distill_require_target_top": True,
    }

    policy, stats = _mcts_distill_policy_target(
        type("Policy", (), {"policy_value_fn": staticmethod(_flat_policy)})(),
        board,
        3,
        config,
    )

    assert stats["accepted"] is True
    assert stats["search_target_top_hit"] is True
    assert policy[3] == 1.0
    assert stats["target_mass"] == 1.0
    assert stats["target_top_hit"] is True


def test_mcts_distill_policy_target_can_reject_weak_search_targets():
    board = Board(width=5, height=5, n_in_row=4)
    board.init_board()
    for move in (0, 1, 2):
        _place(board, move, 1)
    board.current_player = 2
    config = {
        "c_puct": 5,
        "mcts_distill_playouts": 96,
        "mcts_distill_temp": 1.0,
        "mcts_distill_require_target_top": True,
    }

    policy, stats = _mcts_distill_policy_target(
        type("Policy", (), {"policy_value_fn": staticmethod(_bad_block_policy)})(),
        board,
        3,
        config,
    )

    assert policy is None
    assert stats["accepted"] is False
    assert stats["search_target_top_hit"] is False


def test_mcts_distill_data_smoke_with_small_board_policy_net():
    config = {
        "board_width": 5,
        "board_height": 5,
        "n_in_row": 4,
        "mcts_distill_source": "hard_position",
        "mcts_distill_playouts": 2,
        "mcts_distill_temp": 1.0,
        "mcts_distill_target_blend": 0.25,
        "hard_position_noise_stones": 0,
        "hard_position_fork_value": 0.9,
    }
    policy_value_net = PolicyValueNet(
        5,
        5,
        use_gpu=False,
        num_res_blocks=1,
        num_filters=8,
    )

    data, stats = _mcts_distill_data(policy_value_net, config, seed=3, count=2)

    assert len(data) == 2
    assert stats["mcts_distill_samples"] == 2
    assert 0.0 <= stats["mcts_distill_target_mass"] <= 1.0
    for state, policy, value in data:
        assert state.shape == (4, 5, 5)
        assert policy.shape == (25,)
        assert abs(policy.sum() - 1.0) < 1e-6
        assert value == 0.9


def test_mcts_distill_source_config_applies_per_source_overrides():
    config = {
        "mcts_distill_target_mode": "visits",
        "mcts_distill_require_target_top": True,
        "mcts_distill_source_overrides": {
            "threat_space": {
                "mcts_distill_target_mode": "teacher",
                "mcts_distill_require_target_top": False,
            }
        },
    }

    source_config = _mcts_distill_source_config(config, "threat_space")

    assert source_config["mcts_distill_target_mode"] == "teacher"
    assert source_config["mcts_distill_require_target_top"] is False
    assert config["mcts_distill_target_mode"] == "visits"


def test_mcts_distill_data_can_balance_hard_and_threat_sources():
    config = {
        "board_width": 7,
        "board_height": 7,
        "n_in_row": 5,
        "mcts_distill_playouts": 1,
        "mcts_distill_temp": 1.0,
        "mcts_distill_target_mode": "teacher",
        "mcts_distill_value_loss_weight": 0.0,
        "mcts_distill_source_counts": {
            "hard_position": 1,
            "threat_space": 1,
        },
        "mcts_distill_source_overrides": {
            "hard_position": {
                "mcts_distill_require_target_top": False,
                "mcts_distill_min_target_mass": 0.0,
            },
            "threat_space": {
                "mcts_distill_require_target_top": False,
                "mcts_distill_min_target_mass": 0.0,
            },
        },
        "hard_position_noise_stones": 0,
        "hard_position_fork_value": 0.9,
        "threat_space_noise_stones": 0,
        "threat_space_value": 0.8,
    }
    policy_value_net = PolicyValueNet(
        7,
        7,
        use_gpu=False,
        num_res_blocks=1,
        num_filters=8,
    )

    data, stats = _mcts_distill_data(policy_value_net, config, seed=5, count=2)

    assert len(data) == 2
    assert stats["mcts_distill_sources"]["hard_position"] == 1
    assert stats["mcts_distill_sources"]["threat_space"] == 1
    assert stats["mcts_distill_source_stats"]["hard_position"]["samples"] == 1
    assert stats["mcts_distill_source_stats"]["threat_space"]["samples"] == 1
    assert {sample[2] for sample in data} == {0.8, 0.9}


def test_threat_space_proof_value_data_adds_defender_and_followup_values():
    config = {
        "board_width": 7,
        "board_height": 7,
        "n_in_row": 5,
        "threat_space_noise_stones": 0,
        "threat_space_value": 0.8,
        "threat_space_reply_limit": 6,
        "threat_space_followup_limit": 12,
        "threat_space_proof_reply_limit": 3,
        "threat_space_proof_followup_limit": 12,
        "threat_space_proof_root_value": 0.8,
        "threat_space_proof_defender_value": -0.8,
        "threat_space_proof_followup_value": 0.95,
    }

    data, stats = _threat_space_proof_value_data(config, seed=11, count=2)
    values = [sample[2] for sample in data]

    assert stats["threat_space_proof_roots"] == 2
    assert stats["threat_space_proof_defender_states"] == 2
    assert stats["threat_space_proof_followups"] > 0
    assert 0.8 in values
    assert -0.8 in values
    assert 0.95 in values


def test_conversion_teacher_data_extracts_forcing_moves():
    config = {
        "board_width": 7,
        "board_height": 7,
        "n_in_row": 5,
    }
    moves = [1, 20, 2, 21, 3, 22, 4]

    data = _conversion_teacher_data(config, moves, value=0.93)

    assert len(data) == 1
    _state, policy, value = data[0]
    assert policy[4] == 1.0
    assert policy.sum() == 1.0
    assert value == 0.93


def test_conversion_teacher_data_can_extract_two_ply_threats():
    config = {
        "board_width": 7,
        "board_height": 7,
        "n_in_row": 5,
        "conversion_teacher_depth": "one_or_two_ply",
        "conversion_teacher_reply_limit": 8,
        "conversion_teacher_followup_limit": 16,
    }
    moves = [15, 0, 17, 6, 9, 42, 23, 48, 16]

    data = _conversion_teacher_data(config, moves, value=0.85)

    assert len(data) == 1
    _state, policy, value = data[0]
    assert policy[16] == 1.0
    assert policy.sum() == 1.0
    assert value == 0.85


def test_tactical_puzzle_defensive_values_are_configurable():
    config = {
        "board_width": 16,
        "board_height": 16,
        "n_in_row": 5,
        "tactical_block_win_value": 0.65,
        "tactical_block_threat_value": 0.50,
    }

    _state, _policy, block_win_value = _tactical_puzzle_sample(
        config,
        random.Random(1),
        1,
    )
    _state, _policy, block_threat_value = _tactical_puzzle_sample(
        config,
        random.Random(1),
        3,
    )

    assert block_win_value == 0.65
    assert block_threat_value == 0.50


def test_tactical_puzzle_focus_can_emphasize_attack_patterns():
    config = {
        "board_width": 16,
        "board_height": 16,
        "n_in_row": 5,
        "tactical_puzzle_focus": "win_conversion",
        "tactical_attack_threat_value": 0.91,
        "tactical_block_win_value": 0.13,
    }

    _state, _policy, focused_value = _tactical_puzzle_sample(
        config,
        random.Random(1),
        1,
    )

    assert focused_value == 0.91


def test_fork_threat_puzzle_creates_two_open_fours():
    config = {
        "board_width": 16,
        "board_height": 16,
        "n_in_row": 5,
        "hard_position_noise_stones": 0,
    }
    board, target_move, directions = _fork_threat_puzzle_board(
        config,
        random.Random(7),
        0,
    )
    current = board.get_current_player()

    open_four_count = sum(
        1
        for dx, dy in directions
        if line_shape(board, target_move, current, dx, dy) == (4, 2)
    )

    assert target_move in board.availables
    assert open_four_count == 2
    assert fork_threat_count(board, target_move, current, OPEN_FOUR_SCORE) == 2
    assert creates_unanswerable_threat(board, target_move, current)


def test_forcing_win_solver_distinguishes_open_and_closed_fours():
    board = Board(width=7, height=7, n_in_row=5)
    board.init_board()
    for move in (1, 2, 3):
        _place(board, move, 1)
    board.current_player = 1

    assert creates_unanswerable_threat(board, 4, 1)
    assert best_forcing_win_move(board, 1) == 4

    edge_board = Board(width=7, height=7, n_in_row=5)
    edge_board.init_board()
    for move in (0, 1, 2):
        _place(edge_board, move, 1)
    edge_board.current_player = 1

    assert not creates_unanswerable_threat(edge_board, 3, 1)


def test_bounded_two_ply_threat_detects_cross_threat_space():
    board = Board(width=7, height=7, n_in_row=5)
    board.init_board()
    for move in (15, 17, 9, 23):
        _place(board, move, 1)
    board.current_player = 1

    target = 16

    assert not creates_unanswerable_threat(board, target, 1)
    assert creates_bounded_two_ply_threat(
        board,
        target,
        1,
        max_replies=8,
        max_followups=16,
    )
    assert best_bounded_two_ply_threat_move(
        board,
        1,
        max_candidates=16,
        max_replies=8,
        max_followups=16,
    ) == target
    assert best_tactical_move(
        board,
        two_ply_threats=True,
        two_ply_max_candidates=16,
        two_ply_max_replies=8,
        two_ply_max_followups=16,
    ) == target


def test_mcts_can_force_bounded_two_ply_threats():
    board = Board(width=7, height=7, n_in_row=5)
    board.init_board()
    for move in (15, 17, 9, 23):
        _place(board, move, 1)
    board.current_player = 1

    player = MCTSPlayer(
        _flat_policy,
        n_playout=1,
        use_parallel=False,
        tactical_threshold=WIN_SCORE,
        two_ply_threats=True,
        two_ply_max_candidates=16,
        two_ply_max_replies=8,
        two_ply_max_followups=16,
    )
    move, probs = player.get_action(board, return_prob=1)

    assert move == 16
    assert probs[16] == 1.0
    assert player.two_ply_threat_moves == 1
    assert player.search_moves == 0


def test_mcts_uses_threat_solver_even_with_high_shape_threshold():
    board = Board(width=7, height=7, n_in_row=5)
    board.init_board()
    for move in (1, 2, 3):
        _place(board, move, 1)
    board.current_player = 1

    player = MCTSPlayer(
        _flat_policy,
        n_playout=1,
        use_parallel=False,
        tactical_threshold=WIN_SCORE,
    )

    move, probs = player.get_action(board, return_prob=1)

    assert move == 4
    assert probs[4] == 1.0
    assert player.forced_tactical_moves == 1
    assert player.threat_solver_moves == 1


def test_hard_position_puzzle_sample_labels_fork_target():
    config = {
        "board_width": 16,
        "board_height": 16,
        "n_in_row": 5,
        "hard_position_fork_value": 0.97,
        "hard_position_noise_stones": 0,
    }

    _state, policy, value = _hard_position_puzzle_sample(
        config,
        random.Random(9),
        0,
    )

    assert policy.sum() == 1.0
    assert value == 0.97


def test_threat_space_puzzle_board_creates_bounded_two_ply_target():
    config = {
        "board_width": 16,
        "board_height": 16,
        "n_in_row": 5,
        "threat_space_noise_stones": 0,
        "threat_space_reply_limit": 8,
        "threat_space_followup_limit": 16,
    }

    board, target_move, _directions = _threat_space_puzzle_board(
        config,
        random.Random(3),
        0,
    )
    current = board.get_current_player()

    assert target_move in board.availables
    assert not creates_unanswerable_threat(board, target_move, current)
    assert creates_bounded_two_ply_threat(
        board,
        target_move,
        current,
        max_replies=8,
        max_followups=16,
    )


def test_threat_space_puzzle_sample_labels_target_value():
    config = {
        "board_width": 16,
        "board_height": 16,
        "n_in_row": 5,
        "threat_space_value": 0.83,
        "threat_space_noise_stones": 0,
    }

    _state, policy, value = _threat_space_puzzle_sample(
        config,
        random.Random(5),
        1,
    )

    assert policy.sum() == 1.0
    assert value == 0.83


def test_tactical_beam_can_prioritize_fork_threats():
    config = {
        "board_width": 16,
        "board_height": 16,
        "n_in_row": 5,
        "hard_position_noise_stones": 0,
    }
    board, _target_move, _directions = _fork_threat_puzzle_board(
        config,
        random.Random(11),
        0,
    )
    policy_value_net = PolicyValueNet(
        16,
        16,
        use_gpu=False,
        num_res_blocks=1,
        num_filters=8,
        architecture="conv_attention",
    )
    player = TacticalBeamSelfPlayPlayer(
        policy_value_net,
        tactical_guard=False,
        beam_width=8,
        policy_top_k=1,
        value_weight=0.0,
        policy_weight=0.0,
        tactical_weight=0.0,
        fork_weight=5.0,
        noise_frac=0.0,
        seed=3,
    )

    move, probs = player.get_action(board, temp=1e-6, return_prob=1)

    assert fork_threat_count(board, move, board.get_current_player(), OPEN_FOUR_SCORE) >= 2
    assert player.fork_moves == 1
    assert probs[move] == 1.0


def test_self_play_draw_value_relabels_draw_targets_only():
    data = [
        ("state-a", "policy-a", 0.0),
        ("state-b", "policy-b", 0.0),
    ]

    relabeled = _apply_self_play_draw_value(data, -1, -0.1)
    decisive = _apply_self_play_draw_value(data, 1, -0.1)

    assert [item[2] for item in relabeled] == [-0.1, -0.1]
    assert decisive == data


def test_policy_tactical_player_takes_forced_move():
    board = Board(width=5, height=5, n_in_row=4)
    board.init_board()
    _place(board, 0, 1)
    _place(board, 1, 1)
    _place(board, 2, 1)
    board.current_player = 1
    policy_value_net = PolicyValueNet(
        5,
        5,
        use_gpu=False,
        num_res_blocks=1,
        num_filters=8,
        architecture="conv_attention",
    )
    player = PolicyTacticalSelfPlayPlayer(policy_value_net, seed=1)

    move, probs = player.get_action(board, return_prob=1)

    assert move == 3
    assert probs[3] == 1.0
    assert player.forced_tactical_moves == 1
    assert player.policy_moves == 0


def test_policy_tactical_self_play_smoke():
    board = Board(width=5, height=5, n_in_row=4)
    game = Game(board)
    policy_value_net = PolicyValueNet(
        5,
        5,
        use_gpu=False,
        num_res_blocks=1,
        num_filters=8,
        architecture="conv_attention",
    )
    player = PolicyTacticalSelfPlayPlayer(
        policy_value_net,
        seed=2,
        noise_frac=0.0,
    )

    winner, data, moves = game.start_self_play(player, is_shown=0, temp=1.0)

    assert winner in (-1, 1, 2)
    assert len(list(data)) == len(moves)
    assert player.forced_tactical_moves + player.policy_moves == len(moves)
    assert len(moves) > 0


def test_self_play_temperature_schedule_is_applied_per_move():
    class RecordingPlayer:
        def __init__(self):
            self.temps = []

        def set_player_ind(self, player):
            self.player = player

        def reset_player(self):
            pass

        def get_action(self, board, temp=1e-3, return_prob=0):
            self.temps.append(temp)
            move = board.availables[0]
            probs = [0.0] * (board.width * board.height)
            probs[move] = 1.0
            return (move, probs) if return_prob else move

    board = Board(width=3, height=3, n_in_row=3)
    game = Game(board)
    player = RecordingPlayer()
    schedule = _self_play_temperature({
        "temp": 1.0,
        "self_play_temp_cutoff": 2,
        "self_play_late_temp": 0.2,
    })

    _winner, _data, moves = game.start_self_play(player, is_shown=0, temp=schedule)

    assert len(moves) >= 3
    assert player.temps[:3] == [1.0, 1.0, 0.2]


def test_mcts_self_play_dirichlet_noise_can_stop_after_opening():
    board = Board(width=5, height=5, n_in_row=4)
    board.init_board()
    player = MCTSPlayer(
        _flat_policy,
        n_playout=2,
        is_selfplay=1,
        use_parallel=False,
        tactical_threshold=WIN_SCORE,
        dirichlet_frac=0.25,
        dirichlet_moves=1,
    )

    move, _probs = player.get_action(board, temp=1.0, return_prob=1)
    board.do_move(move)
    player.get_action(board, temp=1.0, return_prob=1)

    assert player.dirichlet_noise_moves == 1
    assert player.no_noise_moves == 1


def test_tactical_beam_player_takes_forced_move():
    board = Board(width=5, height=5, n_in_row=4)
    board.init_board()
    _place(board, 0, 1)
    _place(board, 1, 1)
    _place(board, 2, 1)
    board.current_player = 1
    policy_value_net = PolicyValueNet(
        5,
        5,
        use_gpu=False,
        num_res_blocks=1,
        num_filters=8,
        architecture="conv_attention",
    )
    player = TacticalBeamSelfPlayPlayer(policy_value_net, seed=1)

    move, probs = player.get_action(board, return_prob=1)

    assert move == 3
    assert probs[3] == 1.0
    assert player.forced_tactical_moves == 1
    assert player.beam_moves == 0


def test_tactical_beam_player_evaluates_candidates():
    board = Board(width=5, height=5, n_in_row=4)
    board.init_board()
    _place(board, 12, 1)
    board.current_player = 2
    policy_value_net = PolicyValueNet(
        5,
        5,
        use_gpu=False,
        num_res_blocks=1,
        num_filters=8,
        architecture="conv_attention",
    )
    player = TacticalBeamSelfPlayPlayer(
        policy_value_net,
        beam_width=4,
        policy_top_k=4,
        noise_frac=0.0,
        seed=2,
    )

    move, probs = player.get_action(board, return_prob=1)

    assert move in board.availables
    assert player.beam_moves == 1
    assert player.candidate_evaluations > 0
    assert abs(probs.sum() - 1.0) < 1e-6
    assert (probs > 0).sum() <= 8


def test_best_compatible_checkpoint_respects_architecture(tmp_path):
    registry_path = tmp_path / "registry.json"
    register_checkpoint(
        model_path=str(tmp_path / "residual.model"),
        registry_path=registry_path,
        board_width=16,
        board_height=16,
        n_in_row=5,
        num_res_blocks=2,
        num_filters=32,
        architecture="residual",
        games_trained=10,
        elo=1100,
    )
    attention = register_checkpoint(
        model_path=str(tmp_path / "attention.model"),
        registry_path=registry_path,
        board_width=16,
        board_height=16,
        n_in_row=5,
        num_res_blocks=2,
        num_filters=32,
        architecture="conv_attention",
        games_trained=5,
        elo=1000,
    )

    selected = best_compatible_model_checkpoint(
        {
            "board_width": 16,
            "board_height": 16,
            "n_in_row": 5,
            "num_res_blocks": 2,
            "num_filters": 32,
            "architecture": "conv_attention",
        },
        registry_path=registry_path,
    )

    assert selected["id"] == attention["id"]


def test_best_compatible_checkpoint_prefers_stronger_promoted_checkpoint(tmp_path):
    registry_path = tmp_path / "registry.json"
    promotion = {
        "promotion": {
            "promoted": True,
            "promotion_checks": {
                "runtime": True,
                "heuristic_games": True,
                "heuristic_score": True,
                "previous_best_games": True,
                "previous_best_score": True,
                "elo_floor": True,
                "improvement": True,
            },
        },
        "history": {"training": [{"loss": 1.0}]},
    }
    older = register_checkpoint(
        model_path=str(tmp_path / "old_high_elo.model"),
        registry_path=registry_path,
        board_width=16,
        board_height=16,
        n_in_row=5,
        num_res_blocks=4,
        num_filters=64,
        architecture="conv_attention",
        games_trained=2,
        elo=1113,
        metrics=promotion,
    )
    register_checkpoint(
        model_path=str(tmp_path / "new_low_elo.model"),
        registry_path=registry_path,
        board_width=16,
        board_height=16,
        n_in_row=5,
        num_res_blocks=4,
        num_filters=64,
        architecture="conv_attention",
        games_trained=4,
        elo=1092,
        metrics=promotion,
    )

    selected = best_compatible_model_checkpoint(
        {
            "board_width": 16,
            "board_height": 16,
            "n_in_row": 5,
            "num_res_blocks": 4,
            "num_filters": 64,
            "architecture": "conv_attention",
        },
        registry_path=registry_path,
    )

    assert selected["id"] == older["id"]


def test_current_promotion_requires_elo_floor_check(tmp_path):
    registry_path = tmp_path / "registry.json"
    checkpoint = register_checkpoint(
        model_path=str(tmp_path / "stale_promotion.model"),
        registry_path=registry_path,
        board_width=16,
        board_height=16,
        n_in_row=5,
        num_res_blocks=4,
        num_filters=64,
        architecture="conv_attention",
        games_trained=4,
        elo=1120,
        metrics={
            "promotion": {
                "promoted": True,
                "promotion_checks": {
                    "runtime": True,
                    "heuristic_games": True,
                    "heuristic_score": True,
                    "previous_best_games": True,
                    "previous_best_score": True,
                    "improvement": True,
                },
            },
        },
    )

    assert not has_current_promotion(checkpoint)


def test_mcts_self_play_smoke():
    n = 4
    width, height = 5, 5
    model_file = None
    
    board = Board(width=width, height=height, n_in_row=n)
    game = Game(board)

    policy_value_net = PolicyValueNet(
        width,
        height,
        model_file,
        use_gpu=False,
        num_res_blocks=1,
        num_filters=16,
    )
    mcts_player = MCTSPlayer(
        policy_value_net.policy_value_fn,
        c_puct=5,
        n_playout=4,
        use_parallel=False,
    )

    winner, data, moves = game.start_self_play(mcts_player, is_shown=0, temp=1e-3)

    assert winner in (-1, 1, 2)
    assert len(list(data)) == len(moves)
    assert len(moves) > 0

if __name__ == '__main__':
    test_mcts_self_play_smoke()
