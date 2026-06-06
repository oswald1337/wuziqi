import numpy as np

from train import (
    _apply_self_play_draw_value,
    _apply_self_play_target_transform,
    _self_play_target_quality_fields,
)


def _sample(policy, value=1.0):
    state = np.zeros((4, 2, 2), dtype=np.float32)
    return state, np.asarray(policy, dtype=np.float64), float(value)


def test_self_play_target_quality_fields_report_policy_sharpness():
    play_data = [
        _sample([1.0, 0.0, 0.0, 0.0], 1.0),
        _sample([0.25, 0.25, 0.25, 0.25], -1.0),
        _sample([0.0, 0.0, 1.0, 0.0], 1.0),
    ]

    fields = _self_play_target_quality_fields(play_data)

    assert fields["policy_target_samples"] == 3
    assert fields["policy_target_invalid_samples"] == 0
    assert fields["policy_target_one_hot_fraction"] == 2 / 3
    assert fields["policy_target_sharp_fraction"] == 1.0
    assert fields["policy_target_diffuse_fraction"] == 0.0
    assert fields["value_target_positive_fraction"] == 2 / 3
    assert fields["value_target_negative_fraction"] == 1 / 3
    assert fields["value_target_draw_fraction"] == 0.0


def test_draw_value_rewrite_is_reflected_in_target_quality_fields():
    play_data = [
        _sample([1.0, 0.0, 0.0, 0.0], 1.0),
        _sample([0.0, 1.0, 0.0, 0.0], -1.0),
    ]
    rewritten = _apply_self_play_draw_value(play_data, winner=-1, draw_value=-0.12)

    fields = _self_play_target_quality_fields(rewritten)

    assert fields["value_target_negative_fraction"] == 1.0
    assert fields["value_target_draw_fraction"] == 0.0


def test_self_play_target_transform_top_k_sharpens_replay_targets():
    play_data = [
        _sample([0.4, 0.3, 0.2, 0.1], 1.0),
        _sample([0.25, 0.25, 0.25, 0.25], -1.0),
    ]

    transformed, fields = _apply_self_play_target_transform(
        play_data,
        {
            "self_play_target_transform": "top_k",
            "self_play_target_top_k": 2,
        },
    )

    assert fields["policy_target_transform"] == "top_k_2"
    assert fields["policy_target_transform_active"] == 1.0
    assert fields["policy_target_transform_retained_mass_mean"] < 1.0
    assert fields["policy_target_transform_support_kept_fraction_mean"] == 0.5
    assert fields["policy_target_transform_changed_top1_fraction"] == 0.0
    assert fields["policy_target_transform_max_prob_delta"] > 0.0
    assert fields["policy_target_transform_normalized_entropy_delta"] < 0.0
    assert np.allclose(transformed[0][1], [4 / 7, 3 / 7, 0.0, 0.0])
    assert np.allclose(transformed[1][1], [0.5, 0.5, 0.0, 0.0])


def test_self_play_target_transform_identity_keeps_targets_unchanged():
    play_data = [_sample([0.4, 0.3, 0.2, 0.1], 1.0)]

    transformed, fields = _apply_self_play_target_transform(
        play_data,
        {"self_play_target_transform": "identity"},
    )

    assert fields["policy_target_transform"] == "identity"
    assert fields["policy_target_transform_active"] == 0.0
    assert np.array_equal(transformed[0][1], play_data[0][1])
