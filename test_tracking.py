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
        "batched_policy_batches": 4,
        "batched_policy_positions": 512,
        "effective_mcts_batch_size": 64,
        "policy_target_samples": 256,
        "policy_target_max_prob_mean": 0.147,
        "policy_target_entropy_mean": 4.4,
        "policy_target_normalized_entropy_mean": 0.795,
        "policy_target_diffuse_fraction": 0.84,
        "policy_target_sharp_fraction": 0.138,
        "policy_target_one_hot_fraction": 0.138,
        "policy_target_transform_active": 1.0,
        "policy_target_transform_retained_mass_mean": 0.248,
        "policy_target_transform_support_kept_fraction_mean": 0.14,
        "policy_target_transform_changed_top1_fraction": 0.0,
        "policy_target_transform_max_prob_delta": 0.06,
        "policy_target_transform_normalized_entropy_delta": -0.36,
        "value_target_draw_fraction": 0.0,
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
    assert self_play_scalars["self_play/batched_policy_batches"] == 4
    assert self_play_scalars["self_play/batched_policy_positions"] == 512
    assert self_play_scalars["self_play/effective_mcts_batch_size"] == 64
    assert self_play_scalars["self_play/policy_target_samples"] == 256
    assert self_play_scalars["self_play/policy_target_max_prob_mean"] == 0.147
    assert self_play_scalars["self_play/policy_target_entropy_mean"] == 4.4
    assert self_play_scalars["self_play/policy_target_normalized_entropy_mean"] == 0.795
    assert self_play_scalars["self_play/policy_target_diffuse_fraction"] == 0.84
    assert self_play_scalars["self_play/policy_target_sharp_fraction"] == 0.138
    assert self_play_scalars["self_play/policy_target_one_hot_fraction"] == 0.138
    assert self_play_scalars["self_play/policy_target_transform_active"] == 1.0
    assert self_play_scalars["self_play/policy_target_transform_retained_mass_mean"] == 0.248
    assert self_play_scalars["self_play/policy_target_transform_support_kept_fraction_mean"] == 0.14
    assert self_play_scalars["self_play/policy_target_transform_changed_top1_fraction"] == 0.0
    assert self_play_scalars["self_play/policy_target_transform_max_prob_delta"] == 0.06
    assert self_play_scalars["self_play/policy_target_transform_normalized_entropy_delta"] == -0.36
    assert self_play_scalars["self_play/value_target_draw_fraction"] == 0.0
    assert eval_scalars["eval/elo"] == 1134
    assert eval_scalars["eval/heuristic_score"] == 0.5625
    assert eval_scalars["eval/previous_best_score"] == 0.59375


def test_tensorboard_scalars_cover_eval_opponent_progress_tags():
    start_event = {
        "event": "evaluation_opponent_start",
        "opponent_key": "previous_best",
        "games": 4,
        "n_playout": 64,
        "opponent_n_playout": 32,
        "parallel_workers": 4,
        "evaluation_elapsed_s": 1.25,
        "games_trained": 28,
    }
    result_event = {
        "event": "evaluation_opponent",
        "opponent_key": "previous_best",
        "games": 4,
        "n_playout": 64,
        "opponent_n_playout": 32,
        "parallel_workers": 4,
        "score": 0.625,
        "wins": 2,
        "draws": 1,
        "losses": 1,
        "failures": 0,
        "duration_s": 12.5,
        "avg_moves": 84.25,
        "win_avg_moves": 72.0,
        "draw_avg_moves": 101.0,
        "loss_avg_moves": 92.0,
        "evaluation_elapsed_s": 20.0,
        "opponents_completed": 3,
        "games_trained": 28,
    }

    start_scalars = tensorboard_scalars(start_event)
    result_scalars = tensorboard_scalars(result_event)

    assert start_scalars["eval/previous_best_started"] == 1.0
    assert start_scalars["eval/previous_best_games"] == 4
    assert start_scalars["eval/previous_best_candidate_playouts"] == 64
    assert start_scalars["eval/previous_best_opponent_playouts"] == 32
    assert result_scalars["eval/previous_best_score"] == 0.625
    assert result_scalars["eval/previous_best_wins"] == 2
    assert result_scalars["eval/previous_best_draws"] == 1
    assert result_scalars["eval/previous_best_losses"] == 1
    assert result_scalars["eval/previous_best_failures"] == 0
    assert result_scalars["eval/previous_best_avg_moves"] == 84.25
    assert result_scalars["eval/previous_best_win_avg_moves"] == 72.0
    assert result_scalars["eval/previous_best_draw_avg_moves"] == 101.0
    assert result_scalars["eval/previous_best_loss_avg_moves"] == 92.0
    assert result_scalars["runtime/eval_previous_best_seconds"] == 12.5
    assert result_scalars["runtime/eval_elapsed_seconds"] == 20.0
    assert result_scalars["eval/opponents_completed"] == 3
    assert tensorboard_step(result_event) == 28


def test_tensorboard_scalars_cover_parallel_coordination_tags():
    event = {
        "event": "parallel_self_play_batch",
        "parallel_workers": 12,
        "parallel_batch_games": 8,
        "parallel_games_per_second": 0.04,
        "parallel_batch_duration_s": 200.0,
        "parallel_wait_duration_s": 55.0,
        "parallel_wait_calls": 16,
        "parallel_wait_seconds_per_call": 3.4375,
        "parallel_ready_events": 12,
        "parallel_ready_pipes": 30,
        "parallel_ready_pipes_per_event": 2.5,
        "parallel_messages": 38,
        "parallel_predict_messages": 30,
        "parallel_game_result_messages": 8,
        "parallel_coalesce_duration_s": 3.0,
        "parallel_coalesce_calls": 7,
        "parallel_coalesce_wait_calls": 9,
        "parallel_coalesce_extra_pipes": 11,
        "parallel_coalesce_extra_pipes_per_call": 11 / 7,
        "parallel_coalesce_empty_waits": 4,
        "parallel_coalesce_empty_wait_fraction": 4 / 9,
        "parallel_coalesce_fraction": 0.015,
        "parallel_payload_build_duration_s": 1.25,
        "parallel_payload_build_fraction": 0.00625,
        "parallel_request_state_bytes": 1152 * 4 * 16 * 16,
        "parallel_request_state_bytes_per_position": 4 * 16 * 16,
        "parallel_request_available_values": 12000,
        "parallel_request_available_values_per_position": 12000 / 1152,
        "parallel_compact_requests": 30,
        "parallel_compact_request_fraction": 1.0,
        "parallel_response_send_duration_s": 0.8,
        "parallel_response_send_fraction": 0.004,
        "parallel_response_build_duration_s": 0.18,
        "parallel_response_build_fraction": 0.0009,
        "parallel_response_pipe_send_duration_s": 0.62,
        "parallel_response_pipe_send_fraction": 0.0031,
        "parallel_response_probability_values": 6400,
        "parallel_response_probability_values_per_request": 6400 / 30,
        "parallel_compact_responses": 30,
        "parallel_compact_response_fraction": 1.0,
        "gpu_inference_requests": 30,
        "gpu_inference_batches": 9,
        "gpu_inference_positions": 1152,
        "gpu_inference_duration_s": 40.0,
        "gpu_inference_positions_per_batch": 128.0,
        "gpu_inference_positions_per_request": 38.4,
        "gpu_inference_batches_per_request": 0.3,
        "gpu_inference_positions_per_second": 28.8,
    }

    scalars = tensorboard_scalars(event)

    assert scalars["self_play/parallel_workers"] == 12
    assert scalars["runtime/parallel_wait_calls"] == 16
    assert scalars["runtime/parallel_wait_seconds_per_call"] == 3.4375
    assert scalars["runtime/parallel_ready_pipes_per_event"] == 2.5
    assert scalars["runtime/parallel_coalesce_seconds"] == 3.0
    assert scalars["runtime/parallel_coalesce_calls"] == 7
    assert scalars["runtime/parallel_coalesce_extra_pipes_per_call"] == 11 / 7
    assert scalars["runtime/parallel_coalesce_empty_wait_fraction"] == 4 / 9
    assert scalars["runtime/parallel_payload_build_seconds"] == 1.25
    assert scalars["runtime/parallel_request_state_bytes_per_position"] == 4 * 16 * 16
    assert scalars["runtime/parallel_request_available_values_per_position"] == 12000 / 1152
    assert scalars["runtime/parallel_compact_request_fraction"] == 1.0
    assert scalars["runtime/parallel_response_send_seconds"] == 0.8
    assert scalars["runtime/parallel_response_build_seconds"] == 0.18
    assert scalars["runtime/parallel_response_pipe_send_seconds"] == 0.62
    assert scalars["runtime/parallel_response_probability_values_per_request"] == 6400 / 30
    assert scalars["runtime/parallel_compact_response_fraction"] == 1.0
    assert scalars["runtime/gpu_inference_positions_per_request"] == 38.4
    assert scalars["runtime/gpu_inference_batches_per_request"] == 0.3


def test_tensorboard_scalars_cover_runtime_budget_dispatch_tags():
    event = {
        "event": "runtime_budget",
        "max_runtime_s": 600.0,
        "runtime_dispatch_margin_s": 90.0,
        "runtime_elapsed_s": 540.0,
        "runtime_remaining_s": 60.0,
        "runtime_budget_exceeded": False,
        "runtime_dispatch_stopped": True,
        "self_play_requested_games": 64,
        "self_play_dispatched_games": 20,
        "self_play_completed_games": 18,
        "self_play_remaining_games": 46,
        "self_play_stopped_early": True,
        "self_play_stream_elapsed_s": 420.0,
        "self_play_dispatch_stop_estimated_game_s": 75.0,
        "total_games": 30,
    }

    scalars = tensorboard_scalars(event)

    assert scalars["runtime/budget_max_seconds"] == 600.0
    assert scalars["runtime/budget_dispatch_margin_seconds"] == 90.0
    assert scalars["runtime/budget_elapsed_seconds"] == 540.0
    assert scalars["runtime/budget_remaining_seconds"] == 60.0
    assert scalars["runtime/budget_exceeded"] == 0.0
    assert scalars["runtime/dispatch_stopped"] == 1.0
    assert scalars["runtime/dispatch_stop_estimated_game_seconds"] == 75.0
    assert scalars["self_play/dispatched_games"] == 20
    assert scalars["self_play/completed_games"] == 18
    assert scalars["self_play/stopped_early"] == 1.0
    assert tensorboard_step(event) == 30
