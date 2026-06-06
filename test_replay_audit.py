import numpy as np
import pytest

from replay_audit import (
    audit_replay_payload,
    build_transform_spec,
    classify_replay_quality,
    probe_policy_target_transforms,
    select_samples,
    summarize_replay_samples,
    transform_policy_target,
)


def _sample(policy, value):
    state = np.zeros((4, 2, 2), dtype=np.float32)
    return state, np.asarray(policy, dtype=np.float64), float(value)


def test_select_samples_supports_tail_head_and_random():
    samples = list(range(10))

    assert select_samples(samples, max_samples=3, strategy="tail") == [7, 8, 9]
    assert select_samples(samples, max_samples=3, strategy="head") == [0, 1, 2]
    assert len(select_samples(samples, max_samples=3, strategy="random", seed=7)) == 3


def test_summarize_replay_samples_reports_value_and_policy_quality():
    samples = [
        _sample([1.0, 0.0, 0.0, 0.0], 1.0),
        _sample([0.25, 0.25, 0.25, 0.25], -1.0),
        _sample([0.01, 0.01, 0.01, 0.97], 0.0),
    ]

    summary = summarize_replay_samples(samples)

    assert summary["samples"] == 3
    assert summary["value_targets"]["positive"] == 1
    assert summary["value_targets"]["negative"] == 1
    assert summary["value_targets"]["draw"] == 1
    assert summary["policy_targets"]["one_hot_fraction"] == 1 / 3
    assert summary["policy_targets"]["sharp_max_prob_ge_0_25_fraction"] == 1.0
    assert summary["policy_targets"]["diffuse_max_prob_le_0_02_fraction"] == 0.0
    assert summary["policy_targets"]["invalid_policy_samples"] == 0


def test_classify_replay_quality_flags_diffuse_policy_targets():
    samples = [
        _sample(np.full(100, 0.01), 1.0),
        _sample(np.full(100, 0.01), -1.0),
    ]
    summary = summarize_replay_samples(samples)

    assessment = classify_replay_quality(summary)

    assert assessment["label"] == "diffuse_policy_targets"
    assert "target sharpness" in assessment["recommendation"]


def test_classify_replay_quality_flags_draw_heavy_value_targets_first():
    samples = [
        _sample([1.0, 0.0, 0.0, 0.0], 0.0),
        _sample([0.0, 1.0, 0.0, 0.0], 0.0),
        _sample([0.0, 0.0, 1.0, 0.0], 1.0),
    ]
    summary = summarize_replay_samples(samples)

    assessment = classify_replay_quality(summary)

    assert assessment["label"] == "draw_heavy_value_targets"


def test_audit_replay_payload_includes_metadata_and_slice_size():
    payload = {
        "version": 1,
        "games_recorded": 5,
        "config": {"preset": "p"},
        "samples": [
            _sample([1.0, 0.0, 0.0, 0.0], 1.0),
            _sample([0.0, 1.0, 0.0, 0.0], -1.0),
            _sample([0.0, 0.0, 1.0, 0.0], 1.0),
        ],
    }

    result = audit_replay_payload(payload, max_samples=2, strategy="tail")

    assert result["total_samples"] == 3
    assert result["audited_samples"] == 2
    assert result["metadata"]["games_recorded"] == 5
    assert result["metadata"]["config"]["preset"] == "p"


def test_power_transform_sharpens_policy_without_changing_top1():
    transformed, detail = transform_policy_target(
        [0.5, 0.25, 0.25, 0.0],
        power=2.0,
    )

    assert transformed.tolist() == pytest.approx([2 / 3, 1 / 6, 1 / 6, 0.0])
    assert detail["retained_mass_before_renorm"] == pytest.approx(1.0)
    assert detail["changed_top1"] is False
    assert detail["transformed_support"] == 3
    assert detail["kl_transformed_to_original"] > 0.0


def test_top_k_transform_reports_retained_mass_and_support():
    transformed, detail = transform_policy_target(
        [0.4, 0.3, 0.2, 0.1],
        top_k=2,
    )

    assert transformed.tolist() == pytest.approx([4 / 7, 3 / 7, 0.0, 0.0])
    assert detail["retained_mass_before_renorm"] == pytest.approx(0.7)
    assert detail["original_support"] == 4
    assert detail["transformed_support"] == 2
    assert detail["support_kept_fraction"] == pytest.approx(0.5)


def test_min_prob_transform_falls_back_to_top1_when_filter_drops_all_actions():
    transformed, detail = transform_policy_target(
        [0.4, 0.3, 0.2, 0.1],
        min_prob=0.9,
    )

    assert transformed.tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert detail["fallback_to_top1"] is True
    assert detail["retained_mass_before_renorm"] == pytest.approx(0.4)


def test_top1_transform_preserves_first_original_top_action_on_ties():
    transformed, detail = transform_policy_target(
        [0.25, 0.25, 0.25, 0.25],
        top_k=1,
    )

    assert transformed.tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert detail["original_top1"] == 0
    assert detail["transformed_top1"] == 0
    assert detail["changed_top1"] is False


def test_probe_policy_target_transforms_reports_transform_deltas():
    samples = [
        _sample([0.4, 0.3, 0.2, 0.1], 1.0),
        _sample([0.25, 0.25, 0.25, 0.25], -1.0),
    ]
    spec = build_transform_spec("top_k_power", top_k=2, power=2.0)

    probe = probe_policy_target_transforms(samples, [spec])
    result = probe["transforms"][0]

    assert result["name"] == "top_k_2_power_2"
    assert result["transformed_samples"] == 2
    assert result["comparison"]["retained_mass_before_renorm"]["mean"] < 1.0
    assert result["comparison"]["max_prob_mean_delta"] > 0.0
    assert result["comparison"]["normalized_entropy_mean_delta"] < 0.0


def test_audit_replay_payload_can_include_transform_probe():
    payload = {
        "samples": [
            _sample([0.4, 0.3, 0.2, 0.1], 1.0),
            _sample([0.25, 0.25, 0.25, 0.25], -1.0),
        ],
    }

    result = audit_replay_payload(
        payload,
        transform_specs=[build_transform_spec("power", power=2.0)],
    )

    probe = result["policy_target_transform_probe"]
    assert probe["transforms"][0]["name"] == "power_2"
    assert probe["transforms"][0]["comparison"]["max_prob_mean_delta"] > 0.0
