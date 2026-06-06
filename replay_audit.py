import argparse
import json
import math
import pickle
import random
from pathlib import Path
from statistics import mean, median

import numpy as np


CONFIG_SUMMARY_KEYS = (
    "preset",
    "board_width",
    "board_height",
    "n_in_row",
    "num_res_blocks",
    "num_filters",
    "architecture",
    "self_play_mode",
    "self_play_games",
    "self_play_parallel_games",
    "n_playout",
    "eval_n_playout",
    "mcts_batch_size",
    "mcts_min_batches_per_search",
    "gpu_inference_max_batch_size",
    "gpu_inference_coalesce_ms",
    "mcts_heuristic_prior_weight",
    "self_play_draw_value",
    "self_play_target_transform",
    "self_play_target_power",
    "self_play_target_top_k",
    "self_play_target_min_prob",
    "batch_size",
    "epochs",
)

TRANSFORM_CHOICES = (
    "identity",
    "power",
    "top_k",
    "min_prob",
    "top_k_power",
    "top1",
    "custom",
)


def load_replay(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def replay_samples(payload):
    if isinstance(payload, dict):
        return list(payload.get("samples", []))
    return list(payload)


def select_samples(samples, max_samples=50000, strategy="tail", seed=0):
    samples = list(samples)
    if max_samples is None or max_samples <= 0 or len(samples) <= max_samples:
        return samples
    if strategy == "head":
        return samples[:max_samples]
    if strategy == "random":
        rng = random.Random(seed)
        return rng.sample(samples, max_samples)
    if strategy != "tail":
        raise ValueError(f"Unsupported sample strategy: {strategy}")
    return samples[-max_samples:]


def _summary(values):
    values = [
        float(value)
        for value in values
        if isinstance(value, (int, float, np.number))
        and math.isfinite(float(value))
    ]
    if not values:
        return None
    ordered = sorted(values)

    def percentile(q):
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * q
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        frac = position - lower
        return ordered[lower] * (1.0 - frac) + ordered[upper] * frac

    return {
        "min": ordered[0],
        "p10": percentile(0.10),
        "median": median(ordered),
        "mean": mean(ordered),
        "p90": percentile(0.90),
        "max": ordered[-1],
    }


def _normalized_policy(policy):
    raw = np.asarray(policy, dtype=np.float64).ravel()
    finite = np.isfinite(raw)
    negative = finite & (raw < 0.0)
    cleaned = np.where(finite & (raw > 0.0), raw, 0.0)
    mass = float(cleaned.sum())
    diagnostics = {
        "policy_sum": mass,
        "policy_nonfinite": int((~finite).sum()),
        "policy_negative": int(negative.sum()),
        "policy_nonzero": int((cleaned > 1e-12).sum()),
    }
    if mass <= 0.0 or raw.size == 0:
        return raw, cleaned, None, diagnostics
    return raw, cleaned, cleaned / mass, diagnostics


def _policy_metrics(policy):
    raw, _cleaned, probs, result = _normalized_policy(policy)
    mass = result["policy_sum"]
    if mass <= 0.0 or raw.size == 0:
        result.update({
            "policy_max_prob": 0.0,
            "policy_entropy": None,
            "policy_normalized_entropy": None,
        })
        return result
    positive = probs[probs > 0.0]
    entropy = float(-np.sum(positive * np.log(positive)))
    max_entropy = math.log(raw.size) if raw.size > 1 else 0.0
    result.update({
        "policy_max_prob": float(np.max(probs)),
        "policy_entropy": entropy,
        "policy_normalized_entropy": (
            None if max_entropy <= 0.0 else entropy / max_entropy
        ),
        "policy_support_ge_1pct": int((probs >= 0.01).sum()),
        "policy_support_ge_2pct": int((probs >= 0.02).sum()),
    })
    return result


def _policy_metrics_from_probs(probs, raw_size=None):
    raw_size = int(raw_size or probs.size)
    mass = float(np.sum(probs))
    result = {
        "policy_sum": mass,
        "policy_nonfinite": 0,
        "policy_negative": 0,
        "policy_nonzero": int((probs > 1e-12).sum()),
    }
    if mass <= 0.0 or raw_size == 0:
        result.update({
            "policy_max_prob": 0.0,
            "policy_entropy": None,
            "policy_normalized_entropy": None,
        })
        return result
    positive = probs[probs > 0.0]
    entropy = float(-np.sum(positive * np.log(positive)))
    max_entropy = math.log(raw_size) if raw_size > 1 else 0.0
    result.update({
        "policy_max_prob": float(np.max(probs)),
        "policy_entropy": entropy,
        "policy_normalized_entropy": (
            None if max_entropy <= 0.0 else entropy / max_entropy
        ),
        "policy_support_ge_1pct": int((probs >= 0.01).sum()),
        "policy_support_ge_2pct": int((probs >= 0.02).sum()),
    })
    return result


def _summarize_policy_stats(policy_stats, sample_count):
    max_probs = [item["policy_max_prob"] for item in policy_stats]
    entropies = [
        item["policy_entropy"]
        for item in policy_stats
        if item["policy_entropy"] is not None
    ]
    normalized_entropies = [
        item["policy_normalized_entropy"]
        for item in policy_stats
        if item["policy_normalized_entropy"] is not None
    ]
    policy_sums = [item["policy_sum"] for item in policy_stats]
    nonzero = [item["policy_nonzero"] for item in policy_stats]
    support_1pct = [item.get("policy_support_ge_1pct") for item in policy_stats]
    support_2pct = [item.get("policy_support_ge_2pct") for item in policy_stats]
    diffuse = sum(1 for value in max_probs if value <= 0.02)
    sharp = sum(1 for value in max_probs if value >= 0.25)
    one_hot = sum(
        1
        for value, count in zip(max_probs, nonzero)
        if value >= 0.999 and count <= 1
    )
    invalid_policy_samples = sum(
        1
        for item in policy_stats
        if item["policy_nonfinite"] or item["policy_negative"] or item["policy_sum"] <= 0.0
    )
    divisor = sample_count or 1
    return {
        "invalid_policy_samples": invalid_policy_samples,
        "max_prob": _summary(max_probs),
        "entropy": _summary(entropies),
        "normalized_entropy": _summary(normalized_entropies),
        "policy_sum": _summary(policy_sums),
        "nonzero_actions": _summary(nonzero),
        "support_ge_1pct": _summary(support_1pct),
        "support_ge_2pct": _summary(support_2pct),
        "diffuse_max_prob_le_0_02_fraction": diffuse / divisor,
        "sharp_max_prob_ge_0_25_fraction": sharp / divisor,
        "one_hot_fraction": one_hot / divisor,
    }


def _format_number_for_name(value):
    text = f"{float(value):g}"
    return text.replace("-", "neg_").replace(".", "_")


def _transform_name(power=1.0, top_k=None, min_prob=None):
    if top_k == 1 and float(power) == 1.0 and min_prob is None:
        return "top1"
    parts = []
    if top_k is not None:
        parts.append(f"top_k_{int(top_k)}")
    if min_prob is not None:
        parts.append(f"min_prob_{_format_number_for_name(min_prob)}")
    if float(power) != 1.0:
        parts.append(f"power_{_format_number_for_name(power)}")
    return "_".join(parts) if parts else "identity"


def _validate_transform_params(power=1.0, top_k=None, min_prob=None):
    if power <= 0.0:
        raise ValueError("power must be > 0")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be > 0")
    if min_prob is not None and min_prob < 0.0:
        raise ValueError("min_prob must be >= 0")


def build_transform_spec(transform, power=2.0, top_k=None, min_prob=None):
    if transform == "identity":
        power, top_k, min_prob = 1.0, None, None
    elif transform == "power":
        top_k, min_prob = None, None
    elif transform == "top_k":
        if top_k is None:
            raise ValueError("--top-k is required for --transform top_k")
        power, min_prob = 1.0, None
    elif transform == "min_prob":
        if min_prob is None:
            raise ValueError("--min-prob is required for --transform min_prob")
        power, top_k = 1.0, None
    elif transform == "top_k_power":
        if top_k is None:
            raise ValueError("--top-k is required for --transform top_k_power")
    elif transform == "top1":
        power, top_k, min_prob = 1.0, 1, None
    elif transform == "custom":
        pass
    else:
        raise ValueError(f"Unsupported transform: {transform}")
    _validate_transform_params(
        power=power,
        top_k=top_k,
        min_prob=min_prob,
    )
    return {
        "name": _transform_name(
            power=power,
            top_k=top_k,
            min_prob=min_prob,
        ),
        "power": float(power),
        "top_k": None if top_k is None else int(top_k),
        "min_prob": None if min_prob is None else float(min_prob),
    }


def default_transform_specs():
    return [
        build_transform_spec("identity"),
        build_transform_spec("power", power=1.5),
        build_transform_spec("power", power=2.0),
        build_transform_spec("power", power=3.0),
        build_transform_spec("top_k", top_k=32),
        build_transform_spec("top_k", top_k=16),
        build_transform_spec("top_k", top_k=8),
        build_transform_spec("top_k_power", top_k=16, power=2.0),
        build_transform_spec("top_k_power", top_k=8, power=2.0),
        build_transform_spec("min_prob", min_prob=0.01),
        build_transform_spec("min_prob", min_prob=0.02),
        build_transform_spec("top1"),
    ]


def _top_k_mask(probs, top_k):
    positive_indices = np.flatnonzero(probs > 0.0)
    if top_k is None or positive_indices.size <= top_k:
        mask = np.zeros_like(probs, dtype=bool)
        mask[positive_indices] = True
        return mask
    order = np.lexsort((positive_indices, -probs[positive_indices]))
    selected = positive_indices[order[:top_k]]
    mask = np.zeros_like(probs, dtype=bool)
    mask[selected] = True
    return mask


def _transform_normalized_policy(
    probs,
    diagnostics,
    raw_size,
    power=1.0,
    top_k=None,
    min_prob=None,
):
    if probs is None:
        detail = {
            **diagnostics,
            "valid": False,
            "retained_mass_before_renorm": 0.0,
            "fallback_to_top1": False,
            "original_top1": None,
            "transformed_top1": None,
            "changed_top1": False,
            "original_support": 0,
            "transformed_support": 0,
            "support_kept_fraction": None,
            "l1_distance": None,
            "kl_transformed_to_original": None,
            "cross_entropy_to_original": None,
            "transformed_policy_metrics": None,
        }
        return np.zeros(raw_size, dtype=np.float64), detail

    mask = probs > 0.0
    if top_k is not None:
        mask &= _top_k_mask(probs, int(top_k))
    if min_prob is not None:
        mask &= probs >= float(min_prob)

    retained_mass = float(probs[mask].sum())
    fallback_to_top1 = False
    original_top1 = int(np.argmax(probs))
    if retained_mass <= 0.0:
        fallback_to_top1 = True
        mask = np.zeros_like(probs, dtype=bool)
        mask[original_top1] = True
        retained_mass = float(probs[original_top1])

    transformed = np.where(mask, probs, 0.0)
    if float(power) != 1.0:
        transformed = np.where(transformed > 0.0, transformed ** float(power), 0.0)
    transformed_mass = float(transformed.sum())
    if transformed_mass <= 0.0:
        detail = {
            **diagnostics,
            "valid": False,
            "retained_mass_before_renorm": retained_mass,
            "fallback_to_top1": fallback_to_top1,
            "original_top1": original_top1,
            "transformed_top1": None,
            "changed_top1": False,
            "original_support": int((probs > 0.0).sum()),
            "transformed_support": 0,
            "support_kept_fraction": 0.0,
            "l1_distance": None,
            "kl_transformed_to_original": None,
            "cross_entropy_to_original": None,
            "transformed_policy_metrics": None,
        }
        return np.zeros(raw_size, dtype=np.float64), detail

    transformed = transformed / transformed_mass
    transformed_top1 = int(np.argmax(transformed))
    original_support = int((probs > 0.0).sum())
    transformed_support = int((transformed > 1e-12).sum())
    positive_transformed = transformed > 0.0
    kl_transformed_to_original = float(np.sum(
        transformed[positive_transformed]
        * (
            np.log(transformed[positive_transformed])
            - np.log(probs[positive_transformed])
        )
    ))
    cross_entropy_to_original = float(
        -np.sum(
            transformed[positive_transformed]
            * np.log(probs[positive_transformed])
        )
    )
    detail = {
        **diagnostics,
        "valid": True,
        "retained_mass_before_renorm": retained_mass,
        "fallback_to_top1": fallback_to_top1,
        "original_top1": original_top1,
        "transformed_top1": transformed_top1,
        "changed_top1": transformed_top1 != original_top1,
        "original_support": original_support,
        "transformed_support": transformed_support,
        "support_kept_fraction": (
            None
            if original_support <= 0
            else transformed_support / original_support
        ),
        "l1_distance": float(np.sum(np.abs(transformed - probs))),
        "kl_transformed_to_original": kl_transformed_to_original,
        "cross_entropy_to_original": cross_entropy_to_original,
        "transformed_policy_metrics": _policy_metrics_from_probs(
            transformed,
            raw_size=raw_size,
        ),
    }
    return transformed, detail


def transform_policy_target(policy, power=1.0, top_k=None, min_prob=None):
    _validate_transform_params(power=power, top_k=top_k, min_prob=min_prob)
    raw, _cleaned, probs, diagnostics = _normalized_policy(policy)
    return _transform_normalized_policy(
        probs,
        diagnostics,
        raw.size,
        power=power,
        top_k=top_k,
        min_prob=min_prob,
    )


def _summary_mean(record, *path):
    value = record
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, (int, float, np.number)):
        return float(value)
    return None


def _summary_delta(after, before, *path):
    after_value = _summary_mean(after, *path)
    before_value = _summary_mean(before, *path)
    if after_value is None or before_value is None:
        return None
    return after_value - before_value


def _prepare_transform_samples(samples):
    prepared = []
    invalid_samples = 0
    for sample in samples:
        if not _valid_sample(sample):
            invalid_samples += 1
            continue
        raw, _cleaned, probs, diagnostics = _normalized_policy(sample[1])
        prepared.append({
            "sample": sample,
            "raw_size": raw.size,
            "probs": probs,
            "diagnostics": diagnostics,
        })
    return prepared, invalid_samples


def summarize_policy_transform(
    samples,
    spec,
    prepared=None,
    invalid_samples=None,
    original_summary=None,
):
    if prepared is None:
        prepared, prepared_invalid_samples = _prepare_transform_samples(samples)
        if invalid_samples is None:
            invalid_samples = prepared_invalid_samples
    elif invalid_samples is None:
        invalid_samples = 0

    if original_summary is None:
        transformable_samples = [
            entry["sample"]
            for entry in prepared
            if entry["probs"] is not None
        ]
        original_summary = summarize_replay_samples(transformable_samples)

    transformed_policy_stats = []
    details = []
    invalid_policy_samples = 0
    for entry in prepared:
        _transformed, detail = _transform_normalized_policy(
            entry["probs"],
            entry["diagnostics"],
            entry["raw_size"],
            power=spec.get("power", 1.0),
            top_k=spec.get("top_k"),
            min_prob=spec.get("min_prob"),
        )
        if not detail["valid"]:
            invalid_policy_samples += 1
            continue
        transformed_policy_stats.append(detail["transformed_policy_metrics"])
        details.append(detail)

    transformed_policy_targets = _summarize_policy_stats(
        transformed_policy_stats,
        len(transformed_policy_stats),
    )
    divisor = len(details) or 1
    comparison = {
        "retained_mass_before_renorm": _summary(
            detail["retained_mass_before_renorm"] for detail in details
        ),
        "original_support": _summary(
            detail["original_support"] for detail in details
        ),
        "transformed_support": _summary(
            detail["transformed_support"] for detail in details
        ),
        "support_kept_fraction": _summary(
            detail["support_kept_fraction"] for detail in details
        ),
        "l1_distance": _summary(detail["l1_distance"] for detail in details),
        "kl_transformed_to_original": _summary(
            detail["kl_transformed_to_original"] for detail in details
        ),
        "cross_entropy_to_original": _summary(
            detail["cross_entropy_to_original"] for detail in details
        ),
        "changed_top1_fraction": (
            sum(1 for detail in details if detail["changed_top1"]) / divisor
        ),
        "fallback_top1_fraction": (
            sum(1 for detail in details if detail["fallback_to_top1"]) / divisor
        ),
        "max_prob_mean_delta": _summary_delta(
            {"policy_targets": transformed_policy_targets},
            original_summary,
            "policy_targets",
            "max_prob",
            "mean",
        ),
        "normalized_entropy_mean_delta": _summary_delta(
            {"policy_targets": transformed_policy_targets},
            original_summary,
            "policy_targets",
            "normalized_entropy",
            "mean",
        ),
        "diffuse_fraction_delta": (
            transformed_policy_targets["diffuse_max_prob_le_0_02_fraction"]
            - original_summary["policy_targets"][
                "diffuse_max_prob_le_0_02_fraction"
            ]
        ),
        "sharp_fraction_delta": (
            transformed_policy_targets["sharp_max_prob_ge_0_25_fraction"]
            - original_summary["policy_targets"][
                "sharp_max_prob_ge_0_25_fraction"
            ]
        ),
    }
    return {
        "name": spec["name"],
        "spec": spec,
        "input_samples": len(samples),
        "invalid_samples": invalid_samples,
        "invalid_policy_samples": invalid_policy_samples,
        "transformed_samples": len(transformed_policy_stats),
        "original_policy_targets": original_summary["policy_targets"],
        "transformed_policy_targets": transformed_policy_targets,
        "comparison": comparison,
    }


def probe_policy_target_transforms(samples, transform_specs=None):
    specs = transform_specs or default_transform_specs()
    prepared, invalid_samples = _prepare_transform_samples(samples)
    transformable_samples = [
        entry["sample"]
        for entry in prepared
        if entry["probs"] is not None
    ]
    original_summary = summarize_replay_samples(transformable_samples)
    return {
        "transforms": [
            summarize_policy_transform(
                samples,
                spec,
                prepared=prepared,
                invalid_samples=invalid_samples,
                original_summary=original_summary,
            )
            for spec in specs
        ],
    }


def _valid_sample(sample):
    try:
        _state, policy, value = sample
    except (TypeError, ValueError):
        return False
    return policy is not None and isinstance(value, (int, float, np.number))


def summarize_replay_samples(samples):
    valid = [sample for sample in samples if _valid_sample(sample)]
    policy_stats = [_policy_metrics(sample[1]) for sample in valid]
    values = [float(sample[2]) for sample in valid]
    sample_count = len(valid)
    positive = sum(1 for value in values if value > 1e-9)
    negative = sum(1 for value in values if value < -1e-9)
    draw = sum(1 for value in values if abs(value) <= 1e-9)
    divisor = sample_count or 1
    return {
        "samples": sample_count,
        "invalid_samples": len(samples) - sample_count,
        "value_targets": {
            "positive": positive,
            "negative": negative,
            "draw": draw,
            "positive_fraction": positive / divisor,
            "negative_fraction": negative / divisor,
            "draw_fraction": draw / divisor,
            "summary": _summary(values),
        },
        "policy_targets": _summarize_policy_stats(policy_stats, sample_count),
    }


def classify_replay_quality(summary):
    value = summary.get("value_targets", {})
    policy = summary.get("policy_targets", {})
    draw_fraction = value.get("draw_fraction", 0.0)
    diffuse_fraction = policy.get("diffuse_max_prob_le_0_02_fraction", 0.0)
    sharp_fraction = policy.get("sharp_max_prob_ge_0_25_fraction", 0.0)
    entropy_mean = (policy.get("normalized_entropy") or {}).get("mean")
    evidence = [
        f"value draw fraction {draw_fraction:.3f}",
        f"policy diffuse<=0.02 fraction {diffuse_fraction:.3f}",
        f"policy sharp>=0.25 fraction {sharp_fraction:.3f}",
    ]
    if entropy_mean is not None:
        evidence.append(f"policy normalized entropy mean {entropy_mean:.3f}")

    if draw_fraction >= 0.30:
        label = "draw_heavy_value_targets"
        recommendation = "Revise draw handling or value targets before scaling."
    elif diffuse_fraction >= 0.50 and sharp_fraction < 0.25:
        label = "diffuse_policy_targets"
        recommendation = (
            "Improve MCTS target sharpness/alignment before heuristic variants or "
            "longer training."
        )
    elif entropy_mean is not None and entropy_mean >= 0.80:
        label = "high_entropy_policy_targets"
        recommendation = "Inspect search target quality; policy targets may be too uniform."
    else:
        label = "replay_targets_usable"
        recommendation = "Replay targets do not show obvious draw-heavy or diffuse-target failure."

    return {
        "label": label,
        "evidence": evidence,
        "recommendation": recommendation,
    }


def summarize_replay_config(config):
    if not isinstance(config, dict):
        return {}
    summary = {
        key: config.get(key)
        for key in CONFIG_SUMMARY_KEYS
        if key in config
    }
    if isinstance(config.get("init_from"), dict):
        init_from = config["init_from"]
        summary["init_from"] = {
            key: init_from.get(key)
            for key in ("id", "elo", "games_trained", "path")
            if key in init_from
        }
    return summary


def audit_replay_payload(
    payload,
    max_samples=50000,
    strategy="tail",
    seed=0,
    transform_specs=None,
):
    samples = replay_samples(payload)
    selected = select_samples(
        samples,
        max_samples=max_samples,
        strategy=strategy,
        seed=seed,
    )
    summary = summarize_replay_samples(selected)
    metadata = {}
    if isinstance(payload, dict):
        metadata = {
            "version": payload.get("version"),
            "games_recorded": payload.get("games_recorded"),
            "config": summarize_replay_config(payload.get("config", {})),
        }
    result = {
        "total_samples": len(samples),
        "audited_samples": len(selected),
        "sample_strategy": strategy,
        "metadata": metadata,
        **summary,
        "replay_quality_assessment": classify_replay_quality(summary),
    }
    if transform_specs:
        result["policy_target_transform_probe"] = probe_policy_target_transforms(
            selected,
            transform_specs=transform_specs,
        )
    return result


def audit_replay_file(
    path,
    max_samples=50000,
    strategy="tail",
    seed=0,
    transform_specs=None,
):
    payload = load_replay(path)
    result = audit_replay_payload(
        payload,
        max_samples=max_samples,
        strategy=strategy,
        seed=seed,
        transform_specs=transform_specs,
    )
    result["path"] = str(path)
    return result


def main():
    parser = argparse.ArgumentParser(description="Audit replay target quality.")
    parser.add_argument("path", help="Replay pickle path")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50000,
        help="Maximum samples to audit; 0 audits all samples",
    )
    parser.add_argument(
        "--strategy",
        choices=("tail", "head", "random"),
        default="tail",
        help="Which replay slice to audit",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random strategy seed")
    parser.add_argument(
        "--probe-transforms",
        action="store_true",
        help="Compare built-in policy target sharpening/filtering transforms",
    )
    parser.add_argument(
        "--transform",
        choices=TRANSFORM_CHOICES,
        help="Probe one policy target transform in addition to the base audit",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=2.0,
        help="Power for --transform power/top_k_power/custom",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Keep only the top K policy actions for top-k transforms",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        help="Keep only actions whose original probability is at least this value",
    )
    args = parser.parse_args()
    transform_specs = []
    if args.probe_transforms:
        transform_specs.extend(default_transform_specs())
    if args.transform:
        try:
            transform_specs.append(build_transform_spec(
                args.transform,
                power=args.power,
                top_k=args.top_k,
                min_prob=args.min_prob,
            ))
        except ValueError as exc:
            parser.error(str(exc))
    print(json.dumps(
        audit_replay_file(
            args.path,
            max_samples=args.max_samples,
            strategy=args.strategy,
            seed=args.seed,
            transform_specs=transform_specs or None,
        ),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
