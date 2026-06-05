from tracking import tensorboard_scalars, tensorboard_step


def test_tensorboard_scalars_cover_required_training_tags():
    train_event = {
        "event": "train_step",
        "loss": 1.2,
        "policy_loss": 0.8,
        "value_loss": 0.4,
        "entropy": 2.3,
        "priority_samples": 64,
        "conversion_replay_samples": 2048,
        "buffer_size": 512,
        "total_games": 7,
    }

    scalars = tensorboard_scalars(train_event)

    assert scalars["train/loss"] == 1.2
    assert scalars["train/policy_loss"] == 0.8
    assert scalars["train/value_loss"] == 0.4
    assert scalars["train/entropy"] == 2.3
    assert scalars["train/priority_samples"] == 64
    assert scalars["replay/samples"] == 512
    assert scalars["replay/conversion_samples"] == 2048
    assert tensorboard_step(train_event) == 7


def test_tensorboard_scalars_separate_conversion_replay_steps():
    conversion_event = {
        "event": "train_step",
        "source": "conversion_replay",
        "loss": 0.9,
        "policy_loss": 0.9,
        "value_loss": 0.7,
        "entropy": 2.1,
        "priority_samples": 0,
        "conversion_replay_samples": 512,
        "total_games": 9,
    }

    scalars = tensorboard_scalars(conversion_event)

    assert scalars["conversion/loss"] == 0.9
    assert scalars["conversion/policy_loss"] == 0.9
    assert scalars["conversion/value_loss"] == 0.7
    assert scalars["conversion/entropy"] == 2.1
    assert scalars["conversion/priority_samples"] == 0.0
    assert "train/loss" not in scalars
    assert scalars["replay/conversion_samples"] == 512


def test_tensorboard_scalars_separate_conversion_teacher_steps():
    teacher_event = {
        "event": "train_step",
        "source": "conversion_teacher",
        "loss": 0.6,
        "policy_loss": 0.5,
        "value_loss": 0.2,
        "entropy": 1.9,
        "conversion_teacher_samples": 128,
        "total_games": 10,
    }

    scalars = tensorboard_scalars(teacher_event)

    assert scalars["conversion_teacher/loss"] == 0.6
    assert scalars["conversion_teacher/policy_loss"] == 0.5
    assert scalars["conversion_teacher/value_loss"] == 0.2
    assert scalars["conversion_teacher/entropy"] == 1.9
    assert "train/loss" not in scalars
    assert scalars["replay/conversion_teacher_samples"] == 128


def test_tensorboard_scalars_separate_threat_space_steps():
    threat_event = {
        "event": "train_step",
        "source": "threat_space_puzzle",
        "loss": 0.7,
        "policy_loss": 0.6,
        "value_loss": 0.1,
        "entropy": 1.8,
        "buffer_size": 2048,
        "total_games": 11,
    }

    scalars = tensorboard_scalars(threat_event)

    assert scalars["threat_space/loss"] == 0.7
    assert scalars["threat_space/policy_loss"] == 0.6
    assert scalars["threat_space/value_loss"] == 0.1
    assert scalars["threat_space/entropy"] == 1.8
    assert "train/loss" not in scalars
    assert scalars["replay/samples"] == 2048


def test_tensorboard_scalars_separate_threat_space_proof_steps():
    proof_event = {
        "event": "train_step",
        "source": "threat_space_proof",
        "loss": 0.4,
        "policy_loss": 5.0,
        "value_loss": 0.4,
        "entropy": 5.1,
        "threat_space_proof_samples": 128,
        "threat_space_proof_roots": 16,
        "threat_space_proof_defender_states": 16,
        "threat_space_proof_replies": 64,
        "threat_space_proof_followups": 60,
        "threat_space_proof_skipped": 1,
        "total_games": 12,
    }

    scalars = tensorboard_scalars(proof_event)

    assert scalars["proof_value/loss"] == 0.4
    assert scalars["proof_value/policy_loss"] == 5.0
    assert scalars["proof_value/value_loss"] == 0.4
    assert scalars["proof_value/samples"] == 128
    assert scalars["proof_value/roots"] == 16
    assert scalars["proof_value/defender_states"] == 16
    assert scalars["proof_value/replies"] == 64
    assert scalars["proof_value/followups"] == 60
    assert scalars["proof_value/skipped"] == 1
    assert "train/loss" not in scalars


def test_tensorboard_scalars_separate_mcts_distill_steps():
    distill_event = {
        "event": "train_step",
        "source": "mcts_distill",
        "loss": 0.8,
        "policy_loss": 0.7,
        "value_loss": 0.2,
        "entropy": 1.7,
        "mcts_distill_samples": 128,
        "mcts_distill_attempts": 160,
        "mcts_distill_skipped": 32,
        "mcts_distill_accept_rate": 0.8,
        "mcts_distill_target_mass": 0.42,
        "mcts_distill_target_top_rate": 0.55,
        "mcts_distill_search_target_mass": 0.36,
        "mcts_distill_search_top_rate": 0.5,
        "mcts_distill_entropy": 2.3,
        "mcts_distill_leaf_evaluations": 11,
        "mcts_distill_source_stats": {
            "hard_position": {
                "samples": 64,
                "attempts": 96,
                "skipped": 32,
                "accept_rate": 2 / 3,
                "target_mass": 1.0,
                "target_top_rate": 1.0,
                "search_target_mass": 0.9,
                "search_top_rate": 0.8,
                "target_entropy": 0.0,
            }
        },
        "total_games": 12,
    }

    scalars = tensorboard_scalars(distill_event)

    assert scalars["mcts_distill/loss"] == 0.8
    assert scalars["mcts_distill/policy_loss"] == 0.7
    assert scalars["mcts_distill/value_loss"] == 0.2
    assert scalars["mcts_distill/entropy"] == 1.7
    assert scalars["mcts_distill/target_entropy"] == 2.3
    assert scalars["replay/mcts_distill_samples"] == 128
    assert scalars["mcts_distill/attempts"] == 160
    assert scalars["mcts_distill/skipped"] == 32
    assert scalars["mcts_distill/accept_rate"] == 0.8
    assert scalars["mcts_distill/target_mass"] == 0.42
    assert scalars["mcts_distill/target_top_rate"] == 0.55
    assert scalars["mcts_distill/search_target_mass"] == 0.36
    assert scalars["mcts_distill/search_top_rate"] == 0.5
    assert scalars["mcts_distill/leaf_evaluations"] == 11
    assert scalars["mcts_distill/hard_position/samples"] == 64
    assert scalars["mcts_distill/hard_position/attempts"] == 96
    assert scalars["mcts_distill/hard_position/skipped"] == 32
    assert scalars["mcts_distill/hard_position/accept_rate"] == 2 / 3
    assert scalars["mcts_distill/hard_position/target_mass"] == 1.0
    assert scalars["mcts_distill/hard_position/target_top_rate"] == 1.0
    assert scalars["mcts_distill/hard_position/search_target_mass"] == 0.9
    assert scalars["mcts_distill/hard_position/search_top_rate"] == 0.8
    assert scalars["mcts_distill/hard_position/target_entropy"] == 0.0
    assert "train/loss" not in scalars


def test_tensorboard_scalars_cover_self_play_and_eval_tags():
    self_play_event = {
        "event": "self_play_game",
        "winner": -1,
        "moves": 256,
        "forced_tactical_moves": 12,
        "threat_solver_moves": 3,
        "two_ply_threat_moves": 2,
        "search_moves": 244,
        "tactical_prior_searches": 240,
        "tactical_prior_two_ply_hits": 12,
        "tactical_leaf_evaluations": 48,
        "tactical_leaf_positive": 40,
        "tactical_leaf_negative": 8,
        "tactical_leaf_win": 5,
        "tactical_leaf_forcing_win": 20,
        "tactical_leaf_two_ply_threat": 15,
        "tactical_leaf_multiple_immediate_losses": 8,
        "dirichlet_noise_moves": 32,
        "no_noise_moves": 224,
        "buffer_size": 1024,
        "total_games": 8,
    }
    eval_event = {
        "event": "evaluation",
        "elo": 1134,
        "games": 16,
        "evaluation": {
            "opponents": {
                "heuristic": {"score": 0.5625},
                "previous_best": {"score": 0.59375},
            }
        },
    }

    self_play_scalars = tensorboard_scalars(self_play_event)
    eval_scalars = tensorboard_scalars(eval_event)

    assert self_play_scalars["self_play/moves"] == 256
    assert self_play_scalars["self_play/draw_rate"] == 1.0
    assert self_play_scalars["self_play/forced_tactical_moves"] == 12
    assert self_play_scalars["self_play/threat_solver_moves"] == 3
    assert self_play_scalars["self_play/two_ply_threat_moves"] == 2
    assert self_play_scalars["self_play/search_moves"] == 244
    assert self_play_scalars["self_play/tactical_prior_searches"] == 240
    assert self_play_scalars["self_play/tactical_prior_two_ply_hits"] == 12
    assert self_play_scalars["self_play/tactical_leaf_evaluations"] == 48
    assert self_play_scalars["self_play/tactical_leaf_positive"] == 40
    assert self_play_scalars["self_play/tactical_leaf_negative"] == 8
    assert self_play_scalars["self_play/tactical_leaf_win"] == 5
    assert self_play_scalars["self_play/tactical_leaf_forcing_win"] == 20
    assert self_play_scalars["self_play/tactical_leaf_two_ply_threat"] == 15
    assert self_play_scalars["self_play/tactical_leaf_multiple_immediate_losses"] == 8
    assert self_play_scalars["self_play/dirichlet_noise_moves"] == 32
    assert self_play_scalars["self_play/no_noise_moves"] == 224
    assert eval_scalars["eval/elo"] == 1134
    assert eval_scalars["eval/heuristic_score"] == 0.5625
    assert eval_scalars["eval/previous_best_score"] == 0.59375
