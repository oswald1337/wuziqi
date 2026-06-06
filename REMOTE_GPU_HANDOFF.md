# Remote GPU Handoff

This file is the bridge between local M5 experimentation and the real remote
GPU training run. Update it whenever a local experiment changes the best known
recipe.

## Purpose

Local Apple Silicon runs are for finding a good starting point:

- seed checkpoint
- model architecture
- training loop recipe
- curriculum and replay policy
- search/action-selection strategy
- logging and dashboard workflow

Remote GPU runs should scale a documented local recipe instead of repeating
small smoke presets.

## Final Server Wrap, 2026-06-06

The rented GPU server was wrapped before expiry after the MVP run and one
focused efficiency pass.

- Final health check passed. PyTorch CUDA reported one NVIDIA GeForce RTX 4060,
  `nvidia-smi` worked, TensorBoard HTTP returned 200, required training and
  resource TensorBoard tags were present, and the top-human preset resolved to
  12 self-play workers plus 12 eval workers on the current 11.52-core cgroup
  quota.
- TensorBoard was running remotely on port `8081`; view it later with local
  port forwarding to `8080`:

```bash
ssh -i ~/.ssh/vast -p 59644 root@74.48.140.178 -L 8080:localhost:8081
```

- At wrap time there was no active training tmux session. GPU utilization was
  0%, memory use was 0 MiB, and only monitor/dashboard sessions remained.
- The full MVP checkpoint
  `large_16x16_top_human_gpu_final_g524_20260606_014356_758050` failed
  promotion and should not replace the champion seed.
- The last smoke checkpoint
  `large_16x16_top_human_gpu_final_g24_20260606_171202_292769` only validated
  compact request instrumentation; it also failed promotion and is not a seed.
- Source code, tests, and docs should be enough to resume development from
  git. Exact training continuation still needs ignored runtime artifacts from
  `checkpoints/`: seed model, optimizer state, registry, useful replay pickle,
  `training_log.jsonl`, and TensorBoard events. The directory was about 1.9 GB
  at wrap time.

Efficiency evidence from the final compact-request smoke:

- Self-play: 12 games, 915 moves, about 6.37 moves/sec, 802 search moves, no
  self-play draws.
- Loss deltas over the smoke: total -0.281, policy -0.234, value -0.047,
  entropy -0.091. That is only a short mechanical signal, not convergence
  proof.
- Runtime: stream elapsed 143.7s, worker search-time sum 1245.3s, worker wait
  57.9s, coalescing 8.8s, payload build 10.7s, training 8.3s, eval 20.6s,
  checkpoint 8.9s, replay save 5.5s, JSONL/TensorBoard logging below 1s.
- Compact response cut response construction from roughly 9.1s to about 1.3s
  in the packed-response smoke; compact request then reduced request payload
  build to 10.7s and response send to 4.1s. This is useful, but worker wait and
  search cadence still dominate.
- Compact requests sent `uint8` state tensors at 1024 bytes/position on 16x16
  boards plus flattened legal move indexes. Compact responses sent flat
  probability/value arrays and avoided parent-side action/probability lists.
- The best current diagnosis is still MCTS/search-coordination-bound rather
  than model-training-bound. GPU utilization is bursty, CPU quota is not fully
  saturated by useful work, and training/logging are comparatively small.

Next development step: optimize MCTS request cadence, coalescing, and batched
inference/search scheduling before trying another heuristic branch. Only after
the search path is using CPU/GPU efficiently should the next long run compare
target-sharpening or value/draw changes.

## Current Local Artifacts

Copy these to the remote server before a serious run:

- current local seed:
  `large_16x16_attention_mcts_draw_penalty_final_g12_20260604_082155_525442`
- `checkpoints/registry.json`
- best checkpoint model from the registry
- matching `*.optimizer.pt` file when available
- architecture-compatible replay buffer, for example
  `checkpoints/replay_16x16_n5_r*_f*.pkl`
- `config.py`
- `GOAL.md`
- `REMOTE_GPU_HANDOFF.md`
- `train.log` and `checkpoints/training_log.jsonl` for context
- `checkpoints/tensorboard/` when local scalar charts exist

## Remote Recipe Template

Record the chosen values before launching:

- checkpoint id:
- preset:
- board: 16x16, five-in-row
- network and architecture:
- search method:
- self-play games:
- playouts:
- batch size:
- replay policy:
- curriculum:
- optimizer and learning rate:
- evaluation games:
- TensorBoard logdir: `checkpoints/tensorboard`
- expected runtime:

## Candidate Remote Commands

Before scaling search, run the local conv-attention policy/tactical smoke
preset to check that the architecture and curriculum move in the right
direction without paying MCTS cost:

```bash
.venv/bin/python main.py --mode improve --presets large_16x16_attention_smoke --rounds 1 --eval-games 4 --previous-best-games 4
```

This smoke command is intentionally below the current 16-game promotion floor.
Use it to check mechanics and logs, not to select a remote seed.

If the smoke seed looks promising, avoid repeating many 2-game smoke
continuations. Use the longer local policy/tactical recipe to gather enough
cheap self-play and curriculum signal before deciding whether to add MCTS:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_policy_local --resume-best
```

After the policy-only local run failed to improve, test the staged-search beam
probe before scaling expensive MCTS:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_beam_smoke --resume-best
```

If split-loss telemetry shows self-play value loss dominating, try the
value-light policy/tactical branch:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_value_light --resume-best
```

If value-light still fails, delay self-play value updates entirely while
keeping curriculum value training:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_delayed_value --resume-best
```

If delayed self-play value is still flat, test balanced tactical value labels
so defensive blocks are no longer trained as near-loss positions:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_balanced_labels --resume-best
```

If the cheap policy/tactical branches stay flat, run a tiny low-playout MCTS
probe with anchor distillation before deciding whether MCTS is worth scaling:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_probe --resume-best
```

If the tiny MCTS probe promotes, use the modest local MCTS scale-up before the
remote GPU run:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_local --resume-best
```

If the modest MCTS scale-up is draw-heavy or fails promotion, test the
aggressive tactical-guard anti-draw branch before spending more playouts:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_aggressive --resume-best
```

If the open-three guard is too noisy, test the middle threshold that only
forces closed-four-or-better threats:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_closed_four --resume-best
```

If threshold tweaks do not solve draw-heavy play, test explicit draw utility
shaping in self-play:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_draw_penalty --resume-best
```

If the draw-penalty probe survives a 16-game confidence check, scale that same
local recipe cautiously before moving it to the remote GPU:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_draw_penalty_scale --resume-best
```

If scale-up regresses, keep the champion seed and test whether the curriculum
needs more attacking win-conversion targets:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_win_conversion --resume-best
```

If single-line win-conversion targets are still draw-heavy, test hard positions
where one move creates two simultaneous open-four threats:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_fork_teacher --resume-best
```

If hard-position data improves heuristic score but not champion head-to-head,
switch self-play to the fork-weighted tactical beam:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_fork_beam --resume-best
```

If beam self-play is decisive but does not transfer into champion strength,
test the cheap forcing-threat solver inside MCTS:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_threat_solver --resume-best
```

If solver self-play is decisive but uniform replay washes it out, test the
recent decisive conversion-replay lane:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_conversion_replay --resume-best
```

If mixed conversion replay overweights noisy values, test a separate
policy-only conversion rehearsal step:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_conversion_policy_replay --resume-best
```

If policy-only replay is clean but still weak, test explicit high-value
teacher labels for solver-proven conversion moves:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_conversion_teacher --resume-best
```

If one-ply conversion labels still fail, test a bounded two-ply threat-space
teacher before scaling the recipe:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_threat_space_teacher --resume-best
```

If runtime two-ply forcing is too expensive, train on generated offline
two-ply threat-space positions while keeping normal MCTS play/eval:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_threat_space_offline --resume-best
```

If offline threat-space improves heuristic score but self-play still draws,
combine it with fork-weighted beam self-play:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_threat_space_fork_beam --resume-best
```

If beam self-play does not transfer to fixed MCTS, keep MCTS self-play and
raise draw pressure instead:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_threat_space_draw_pressure --resume-best
```

If value pressure alone does not move MCTS out of draws, anneal MCTS
self-play temperature/noise so only the opening is exploratory:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_threat_space_temp_anneal --resume-best
```

If annealing changes behavior but not strength, test soft tactical root priors
inside MCTS before trying more expensive search:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_threat_prior_mcts --resume-best
```

If soft priors are not enough, test direct tactical value backups at MCTS
leaves:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_threat_leaf_mcts --resume-best
```

Run the MCTS-aligned tactical distillation probe:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_distill_tactical --resume-best
```

Run the strict accepted-target MCTS distillation probe:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_distill_sharp --resume-best
```

Run the source-balanced threat-space distillation probe:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_distill_balanced_threat --resume-best
```

Run the proof-aware two-ply root-prior probe:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_two_ply_prior_mcts --resume-best
```

Run the threat-space proof-path value-calibration probe:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_threat_value_calibration --resume-best
```

Start from the current best compatible checkpoint:

```bash
.venv/bin/python main.py --mode improve --presets large_16x16_top_human_gpu --rounds 10 --eval-games 32 --previous-best-games 32
```

Run the local Apple Silicon conv-attention scouting preset:

```bash
.venv/bin/python main.py --mode improve --presets large_16x16_attention_mps --rounds 3 --eval-games 24 --previous-best-games 24
```

Run a full completion gate only for a candidate that clearly improves in
shorter evaluations:

```bash
.venv/bin/python main.py --mode evaluate --agent CHECKPOINT_ID --eval-games 200 --previous-best-games 200
```

Run a non-promoting native-policy diagnostic when a branch trains with a
different action-selection stack than the default MCTS evaluator:

```bash
.venv/bin/python main.py --mode evaluate --agent CHECKPOINT_ID --eval-games 2 --previous-best-games 0 --eval-mode native --no-promote
```

Run the dashboard on the remote server and forward the port locally:

```bash
.venv/bin/python main.py --mode web --host 0.0.0.0 --port 8000
```

Run TensorBoard for scalar monitoring on the current remote GPU box:

```bash
.venv/bin/tensorboard --logdir checkpoints/tensorboard --host 0.0.0.0 --port 8081
```

View it locally by forwarding local port `8080` to remote `8081`:

```bash
ssh -i ~/.ssh/vast -p 59644 root@74.48.140.178 -L 8080:localhost:8081
```

Run the lightweight resource monitor beside long GPU jobs:

```bash
.venv/bin/python resource_monitor.py --output resource_monitor.jsonl --interval 30
```

Before starting a long run, use the one-command preflight:

```bash
.venv/bin/python remote_health_check.py --preset large_16x16_top_human_gpu --eval-games 32
```

It checks tmux monitor sessions, TensorBoard HTTP, training/resource
TensorBoard scalar coverage, fresh resource JSONL, `nvidia-smi`, PyTorch CUDA,
and the resolved self-play/eval worker counts for the preset.

`resource_monitor.py` is cgroup-aware. In `resource_monitor.jsonl`,
`cpu_util_percent` measures the training container against its allocated CPU
quota, while `host_cpu_util_percent` is host-wide pressure and can include
other workloads. The `large_16x16_top_human_gpu` preset uses auto parallelism
for self-play and evaluation, capped to the usable cgroup CPU worker count by
default. The monitor also mirrors CPU/GPU resource scalars into
`checkpoints/tensorboard/resource_monitor` unless started with
`--no-tensorboard`.

The resource monitor was started after the 2026-06-06 MVP run, so that run's
CPU/GPU resource evidence still comes from training JSONL plus `gpu_smi.log`.
For the next long run, leave `resource_monitor.py` running through an idle
baseline window and the full training/evaluation window, then pass
`--resource-log resource_monitor.jsonl` and
`--replay-path checkpoints/replay_16x16_n5_r4_f64.pkl` to
`training_audit.py`.

Before changing the MCTS target recipe, audit replay quality:

```bash
.venv/bin/python replay_audit.py checkpoints/replay_16x16_n5_r4_f64.pkl --max-samples 50000 --strategy tail
```

Before training a target-sharpening rewrite, compare candidate transforms on
the same replay slice without changing trainer behavior:

```bash
.venv/bin/python replay_audit.py checkpoints/replay_16x16_n5_r4_f64.pkl --max-samples 50000 --strategy tail --probe-transforms
```

The 2026-06-06 MVP replay tail classified as `diffuse_policy_targets`: 0.0%
draw labels, about 50/50 positive/negative value labels, 84.0% of policy
targets with max probability <= 0.02, and only 13.8% with max probability >=
0.25. That supports fixing search target sharpness/alignment before another
long heuristic experiment.
The transform probe reports retained MCTS mass, support size, top-1 agreement,
max-prob/entropy deltas, and target distortion metrics for power, top-k,
min-prob, top-k+power, and top-1 candidates. Use smaller `--max-samples`
values for quick checks; the 50k source-of-truth replay tail can take a few
minutes because the replay pickle is large.
The selected focused rewrite for the next run is top-k-16 self-play target
sharpening: `self_play_target_transform=top_k` and
`self_play_target_top_k=16` in `large_16x16_top_human_gpu`. The 50k replay
probe showed this raises mean policy max probability from about 0.147 to
0.207, drops normalized entropy by about 0.366, preserves the search top move,
and retains about 24.8% of original MCTS target mass before renormalization.
Future self-play JSONL/TensorBoard events include transform retained mass,
support kept, top-1 change rate, max-prob delta, and normalized-entropy delta.
The integrated run audit now reports
`decision_recommendation.label=fix_search_target_alignment_before_scaling` for
the MVP artifacts.
Runs started after this instrumentation also write
`self_play/policy_target_*` and `self_play/value_target_*` scalars directly to
JSONL/TensorBoard, so target diffusion should be visible during the run.

## Experiment Notes

Use this table for serious local or remote experiments.

| Date | Preset/Branch | Main Change | Best Checkpoint | Result | Next Decision |
| --- | --- | --- | --- | --- | --- |
| 2026-06-04 | `large_16x16_attention_smoke` | Added `conv_attention 4x64` plus `policy_tactical` direct policy self-play with tactical guards. | `large_16x16_attention_smoke_final_g2_20260604_070112_858646` | Promoted smoke seed at Elo 1113; 4-game probe scored 87.5% vs heuristic and 75% vs previous best, but completion gate stayed false. | Keep as best conv-attention seed; do not treat 4-game evidence as real strength. |
| 2026-06-04 | `large_16x16_attention_smoke` x3 continuation | Repeated 2-game policy/tactical continuations from the seed with 8-game probes. | Seed remained best; candidates `...070546`, `...070720`, `...070852` did not promote. | Candidate Elos 1018, 1054, 1039; heuristic scores 50.0%, 62.5%, 43.75%; previous-best scores 31.25%, 37.5%, 43.75%. Replay grew to 15,056 samples, but tiny continuations were noisy/regressive. | Stop repeating smoke loops; try `large_16x16_attention_policy_local` with more curriculum and 16 cheap self-play games before adding expensive MCTS. |
| 2026-06-04 | `large_16x16_attention_policy_local` | Longer CPU-safe conv-attention policy/tactical run from the seed: 32 teacher games, 8192 puzzles, 16 direct-policy self-play games, 4x training epochs, 8-game eval. | Seed still remained best; `large_16x16_attention_policy_local_final_g18_20260604_071351_994123` rejected. | Elo 1039; 8-game promotion probe scored 43.75% vs heuristic and 43.75% vs previous best. Generated 1706 self-play moves with 310 tactical overrides and grew replay to 38,344 samples, but did not translate into stronger play. | Policy-only tactical training is not enough in this form. Next try staged search: supervised/tactical curriculum first, then low-playout MCTS or threat-search/beam hybrid once the policy is less uniform. |
| 2026-06-04 | `large_16x16_attention_beam_smoke` | Staged-search probe: tactical guard plus one-ply policy/value beam over top tactical and policy candidates. | Seed still remained best; `large_16x16_attention_beam_smoke_final_g6_20260604_072212_702031` rejected. | Elo 1055; 8-game probe scored 43.75% vs heuristic and 50.0% vs previous best. Generated 387 self-play moves with 72 tactical overrides, 315 beam moves, and 3798 candidate evaluations; replay grew to 46,264 samples. | One-ply value beam worked mechanically but did not improve strength. Next inspect value targets/draw handling and try stronger tactical outcome labeling or a very small low-playout MCTS stage from the seed. |
| 2026-06-04 | split-loss smoke telemetry | Added `policy_loss`, `value_loss`, and loss weights to trainer logs/registry/dashboard; reran `large_16x16_attention_smoke`. | `large_16x16_attention_smoke_final_g4_20260604_072629_112437` is diagnostic only. | Built-in eval stayed at Elo 1014. Bootstrap value loss was about 0.17, but self-play value loss rose to about 0.82 while policy loss stayed around 5.5. | Value learning and draw/outcome labeling are suspect. Try lower `value_loss_weight`, better terminal/tactical value labels, or delay value-heavy self-play updates until evaluation improves. |
| 2026-06-04 | `large_16x16_attention_value_light` | Diagnostic branch: keep policy/tactical self-play but reduce `value_loss_weight` to 0.25. | Seed still remained best; `large_16x16_attention_value_light_final_g6_20260604_072806_649365` rejected. | First run Elo 1066 with 50.0% vs heuristic and 50.0% vs previous best over 8-game promotion probe. Repeat run built-in eval fell to Elo 1025. Split-loss JSON logging now works; self-play value loss stayed around 0.54-0.86 but only contributes 25% to total loss. | Value-light is less damaging than full-weight value loss, but still not enough. Next likely step is better value labels or delayed value training, then a tiny low-playout MCTS stage from the seed. |
| 2026-06-04 | `large_16x16_attention_delayed_value` | Diagnostic branch: value loss is trained on bootstrap/curriculum, but self-play updates use `self_play_value_loss_weight=0.0`. | Seed still remained best; `large_16x16_attention_delayed_value_final_g6_20260604_073528_470077` rejected. | Elo 1050; 8-game probe scored 50.0% vs heuristic and 43.75% vs previous best. Self-play updates correctly logged `value_loss_weight=0.0`, with 450 self-play moves and 79 tactical overrides. | Delaying self-play value was worse than value-light. The likely issue is policy target quality or tactical/value label design, not just self-play value weighting. Next try stronger tactical target generation or a tiny low-playout MCTS stage from the seed. |
| 2026-06-04 | `large_16x16_attention_balanced_labels` | Diagnostic branch: tactical puzzle value labels are configurable, with defensive immediate blocks raised to 0.65 and defensive threat blocks raised to 0.50; self-play value loss is kept at 0.25 weight. | Seed still remained best; `large_16x16_attention_balanced_labels_final_g6_20260604_074259_519616` rejected. | Elo 1060; 8-game probe scored 56.25% vs heuristic and 43.75% vs previous best. Tactical puzzle value loss dropped to 0.044 with the balanced labels, but self-play value loss still stayed high at about 0.68-0.89. | Balanced labels are mechanically useful but not enough. Next try a tiny low-playout MCTS stage from the seed, then stronger tactical target generation if MCTS remains too expensive or draw-heavy. |
| 2026-06-04 | `large_16x16_attention_mcts_probe` | Diagnostic branch: tiny conv-attention MCTS replay stage with 8 playouts, balanced tactical labels, and 1024 anchor distillation samples. | `large_16x16_attention_mcts_probe_final_g4_20260604_074638_438236` promoted locally. | Elo 1092; 8-game probe scored 56.25% vs heuristic and 56.25% vs previous best with zero failures. Self-play was feasible on CPU: one 256-move draw took 7.42s and one 93-move win took 3.61s; replay hit the 65,536 cap. `--resume-best` now prefers the latest promoted compatible checkpoint, so the modest MCTS local run resumes from this probe rather than the older noisy-Elo seed. | MCTS is worth scaling modestly, but draw-heavy replay is a warning. Next local step should increase games/playouts slowly, keep anchor distillation, and track win rate rather than only Elo. |
| 2026-06-04 | `large_16x16_attention_mcts_local` | Modest CPU MCTS scale-up from the promoted probe: 16 self-play games, 16 playouts, 4096 balanced tactical puzzles, 2048 anchor samples, larger replay. | Champion remains `large_16x16_attention_mcts_probe_final_g4_20260604_074638_438236`; `large_16x16_attention_mcts_local_final_g20_20260604_075752_431052` rejected. | Elo 1069; 8-game probe scored 56.25% vs heuristic and 50.0% vs previous best. Self-play itself was mostly decisive, with 14 decisive games and 2 full-board draws, but evaluation stayed draw-heavy: built-in heuristic eval was 8 draws, promotion eval was 1 win plus 7 draws vs heuristic and 1 win/6 draws/1 loss vs previous best. Replay grew to 84,176 samples. | Do not simply scale 16-playout MCTS yet. Try `large_16x16_attention_mcts_aggressive`, which lowers the tactical guard threshold to open-three strength, then decide whether to add anti-draw value shaping or stronger threat-search data. |
| 2026-06-04 | `large_16x16_attention_mcts_aggressive` | Lowered MCTS tactical guard threshold to open-three strength (`35000`) for both training and evaluation, so the agent forces more threat play. | Champion remains `large_16x16_attention_mcts_probe_final_g4_20260604_074638_438236`; `large_16x16_attention_mcts_aggressive_final_g12_20260604_080456_222733` was promising but rejected after confidence probe. | 8-game probe promoted at Elo 1095, scoring 62.5% vs heuristic and 56.25% vs previous best. A 16-game confidence probe fell below promotion floor: Elo 1093, 50.0% vs heuristic and 53.125% vs previous best. Training self-play had 7 decisive games and 1 draw, but forced tactical moves dominated many games. | Open-three forcing is useful but too noisy/overforced. Next test `large_16x16_attention_mcts_closed_four`, which forces closed-four-or-better threats while leaving open-threes to search. |
| 2026-06-04 | `large_16x16_attention_mcts_closed_four` | Middle tactical-threshold probe: 16-playout MCTS with threshold `120000`, forcing closed-four-or-better threats but not open threes. | Champion remains `large_16x16_attention_mcts_probe_final_g4_20260604_074638_438236`; `large_16x16_attention_mcts_closed_four_final_g12_20260604_081500_789651` rejected. | Elo 1032; 8-game probe scored 43.75% vs heuristic and 43.75% vs previous best. Self-play was 4 decisive games and 4 draws, with the first three games all full-board draws. | Closed-four threshold is worse than both default/probe and open-three. Next try explicit anti-draw value shaping with `large_16x16_attention_mcts_draw_penalty`. |
| 2026-06-04 | `large_16x16_attention_mcts_draw_penalty` | Training-loop anti-draw probe: open-three tactical guard plus self-play draw targets relabeled to `-0.10`. | Current local champion: `large_16x16_attention_mcts_draw_penalty_final_g12_20260604_082155_525442`. | 16-game confidence probe promoted at Elo 1134, scoring 56.25% vs heuristic and 59.375% vs previous best, with zero failures. The result still had many draws: 3W/12D/1L vs heuristic and 4W/11D/1L vs previous best. Self-play generated 2 decisive games and 6 full-board draws, replay reached 131,072 samples, and final split loss was 5.6087 total / 5.4128 policy / 0.3918 value. | Draw penalty is useful, but the draw rate remains the central weakness. A cautious scale-up was tested next and did not promote, so this champion remains the local seed. |
| 2026-06-04 | `large_16x16_attention_mcts_draw_penalty_scale` | Cautious scale-up of the draw-penalty recipe: 16 self-play games, 24-playout search, 6144 tactical puzzles, 3072 anchor samples, and larger replay. | Rejected; champion remains `large_16x16_attention_mcts_draw_penalty_final_g12_20260604_082155_525442`. | Elo 1060; 12-game promotion probe scored 50.0% vs heuristic and 41.67% vs previous best. Self-play looked healthier than the probe, with 11 decisive games and 5 full-board draws, replay grew to 159,344 samples, and final split loss improved to 5.3545 total / 5.1936 policy / 0.3218 value, but evaluation collapsed to all draws vs heuristic and 0W/10D/2L vs the champion. | Do not scale this recipe blindly on GPU. Next branch should improve data quality and win conversion: hard-position replay, stronger threat-search/teacher targets, or an explicit curriculum for converting open-threes/fours into wins before increasing playouts again. |
| 2026-06-04 | `large_16x16_attention_mcts_win_conversion` | Added `tactical_puzzle_focus=win_conversion`, over-sampling attacking win/threat puzzles while keeping the champion's small MCTS/draw-penalty recipe. | Rejected; champion remains `large_16x16_attention_mcts_draw_penalty_final_g12_20260604_082155_525442`. | Elo 1073; 8-game promotion probe scored 50.0% vs heuristic and 50.0% vs previous best. Training self-play was draw-heavy again: 2 decisive games and 6 full-board draws. The built-in eval was 8 draws vs heuristic, and promotion eval was 1W/6D/1L vs the champion. | Merely changing tactical puzzle frequency is not enough. Next branch should create richer hard positions or a threat-search teacher that supplies multi-move conversion targets, then evaluate with the same small gate before spending on more MCTS. |
| 2026-06-04 | `large_16x16_attention_mcts_fork_teacher` | Added `hard_position_puzzles`: generated fork targets where one move creates two simultaneous open-four threats, then trained the same small MCTS/draw-penalty recipe from the champion. | Rejected; champion remains `large_16x16_attention_mcts_draw_penalty_final_g12_20260604_082155_525442`. | Elo 1094; 8-game promotion probe scored 62.5% vs heuristic and 50.0% vs previous best. Hard-position bootstrap trained 4096 fork positions with loss 5.1873 / policy 5.1128 / value 0.0745. Self-play was still draw-heavy at 3 decisive games and 5 full-board draws. Built-in eval was 8 draws vs heuristic, but promotion eval found 2 wins vs heuristic. | Fork-target data is a partial positive signal against heuristic but not enough to beat the champion. Next branch should change action selection so self-play actively creates and converts fork threats, not just recognize them as supervised positions. |
| 2026-06-04 | `large_16x16_attention_fork_beam` | Added TensorBoard tracking, explicit fork-pressure scoring in tactical beam self-play, and a fork-weighted beam preset using hard-position puzzles. | Rejected after confidence; champion remains `large_16x16_attention_mcts_draw_penalty_final_g12_20260604_082155_525442`. | Built-in eval looked promising at Elo 1087 with 68.75% vs heuristic, and self-play was 12 decisive games with 0 draws. The 8-game promotion probe briefly passed at Elo 1089, but the 16-game confidence probe failed promotion: Elo 1132, 56.25% vs heuristic, 53.125% vs previous best, zero failures. A non-promoting native-policy diagnostic finished at Elo 1030 over 2-game baseline probes: 2/2 vs random, 1W/1D vs heuristic, zero failures. TensorBoard events were written under `checkpoints/tensorboard/large_16x16_attention_fork_beam`. | Fork-beam is useful for generating decisive replay and readable draw-rate telemetry, but it does not yet transfer into reliable champion strength. Preserve useful decisive replay, but do not seed remote training from this checkpoint unless a stronger threat-search evaluator or larger native-policy probe beats the draw-penalty champion. |
| 2026-06-04 | `large_16x16_attention_threat_solver` | Added a cheap forcing-win solver to `best_tactical_move`: it detects non-immediate moves that create at least two immediate winning replies while the opponent has no immediate win. MCTS and staged players now log `threat_solver_moves`, mirrored into TensorBoard as `self_play/threat_solver_moves`. | Rejected; champion remains `large_16x16_attention_mcts_draw_penalty_final_g12_20260604_082155_525442`. | Self-play quality improved: 8/8 decisive games, zero full-board draws, and exactly one solver move per self-play game. Built-in eval was Elo 1071 with 3W/4D/1L vs heuristic. The promotion probe failed at Elo 1047: 0W/7D/1L vs heuristic and 0W/7D/1L vs previous best, zero failures. | The solver is useful for decisive replay and scalar diagnostics, but by itself it does not beat the champion under fixed MCTS evaluation. Keep the solver primitive, but the next branch should preserve decisive solver/beam replay while changing training targets or evaluator/search depth, rather than seeding remote training from this checkpoint. |
| 2026-06-04 | `large_16x16_attention_conversion_replay` | Added a recent decisive replay lane mixed into self-play training batches via `conversion_replay_fraction`; TensorBoard now logs `train/priority_samples` and `replay/conversion_samples`. | Rejected; champion remains `large_16x16_attention_mcts_draw_penalty_final_g12_20260604_082155_525442`. | The mechanism worked: self-play updates used 128 priority samples per 256-sample batch, conversion replay reached 2528 samples, and self-play was 7 decisive games plus 1 full-board draw. Built-in eval regressed to Elo 1023 with 43.75% vs heuristic. A 4-game non-promoting diagnostic looked noisy-promising at 75% vs previous best, but the 8-game confidence probe collapsed to Elo 993, 31.25% vs heuristic, and 31.25% vs previous best, zero failures. | Do not use 50% conversion replay as the remote recipe. The likely issue is noisy decisive value targets, not insufficient replay frequency alone. Next local test should lower conversion replay fraction, use policy-only conversion replay, or add teacher/value labels for conversion positions before oversampling them. |
| 2026-06-04 | `large_16x16_attention_conversion_policy_replay` | Kept normal self-play updates, then added a separate policy-only conversion replay step (`value_loss_weight=0.0`) on recent decisive positions. TensorBoard separates these as `conversion/*` scalars. | Rejected; champion remains `large_16x16_attention_mcts_draw_penalty_final_g12_20260604_082155_525442`. | The training mechanics were cleaner: self-play was 8/8 decisive, conversion replay reached 5040 samples, normal self-play value loss stayed around 0.31-0.33 late, and policy-only conversion loss fell to 4.94. Fixed evaluation still failed: built-in Elo 1023 with 43.75% vs heuristic, and a 4-game non-promoting probe scored Elo 1031, 37.5% vs heuristic, 50.0% vs previous best, zero failures. | Policy-only rehearsal avoids the worst value-target blow-up but still does not improve fixed MCTS strength. Next local work should stop tweaking replay frequency alone and add better conversion labels or stronger teacher/search targets before sending this recipe to remote GPU. |
| 2026-06-04 | `large_16x16_attention_conversion_teacher` | Added extraction of solver-proven conversion moves from self-play and relabeled those positions with explicit high-value teacher targets. TensorBoard separates these as `conversion_teacher/*` scalars. | Rejected; champion remains `large_16x16_attention_mcts_draw_penalty_final_g12_20260604_082155_525442`. | Teacher mechanics were coherent: self-play was 8/8 decisive, one teacher position was extracted per game, teacher replay reached 64 augmented samples, teacher policy loss fell from 5.56 to 3.95, and teacher value loss stayed low around 0.05-0.10. Fixed evaluation still regressed: built-in Elo 991 with 31.25% vs heuristic. A 4-game non-promoting diagnostic again showed a noisy 75% vs previous best but only 37.5% vs heuristic, zero failures. | High-value labels for the current one-move forcing conversion are not enough. The next local branch should improve the evaluator/search side or add deeper multi-move threat-space teacher targets, because local replay/label variations keep producing decisive self-play without fixed-evaluator strength. |
| 2026-06-04 | `large_16x16_attention_threat_space_teacher` | Added a bounded two-ply threat-space primitive, optional MCTS two-ply tactical forcing, `conversion_teacher_depth=one_or_two_ply`, and TensorBoard scalar `self_play/two_ply_threat_moves`. | Rejected; champion remains `large_16x16_attention_mcts_draw_penalty_final_g12_20260604_082155_525442`. | The primitive worked mechanically: self-play was 8/8 decisive, every self-play game used one two-ply threat move and one solver move, teacher positions rose to 2 per game, teacher replay reached 128 augmented samples, and teacher policy loss fell from 5.46 to 4.16. Built-in eval was Elo 1055 with 56.25% vs heuristic, but a 4-game no-promote probe scored Elo 1015, 37.5% vs heuristic, and 37.5% vs previous best, zero failures. Evaluation was noticeably slower with two-ply forcing enabled. | Keep the bounded two-ply primitive as a diagnostic/teacher tool, but do not scale this preset directly. Next local work should test cheaper use: teacher-only two-ply labels, capped eval/search usage, or generating offline threat-space puzzle data instead of forcing two-ply checks on every MCTS move. |
| 2026-06-04 | `large_16x16_attention_threat_space_offline` | Added generated offline two-ply threat-space bootstrap positions, TensorBoard `threat_space/*` scalars, and a local preset that keeps `mcts_two_ply_threats=false` for normal MCTS play/eval. | Rejected; champion remains `large_16x16_attention_mcts_draw_penalty_final_g12_20260604_082155_525442`. | The first validating generator pass was too slow locally: 4096 proof-checked positions took about 5 minutes. Switching to the verified construction without per-board validation made 4096 positions train in seconds. Threat-space bootstrap loss was 4.98 with near-zero value loss. Built-in eval reached Elo 1071 and 62.5% vs heuristic. A 4-game no-promote probe scored 75.0% vs heuristic, 50.0% vs previous best, and zero failures, but self-play was still 4 decisive games plus 4 full-board draws. | Offline two-ply data is promising for heuristic conversion and cheap enough after removing per-board validation, but it still does not beat the champion. Next local branch should combine offline threat-space data with a draw-reduction/search change, or run a confidence probe only after previous-best score clears 55%. |
| 2026-06-04 | `large_16x16_attention_threat_space_fork_beam` | Combined generated offline two-ply threat-space curriculum with fork-weighted tactical-beam self-play to test whether decisive replay plus stronger conversion targets transfers into fixed MCTS evaluation. | Rejected; champion remains `large_16x16_attention_mcts_draw_penalty_final_g12_20260604_082155_525442`. | The draw-reduction goal worked in self-play: 12/12 games were decisive, threat solver fired once per game, and late policy loss fell below 4.9. However `fork_moves` stayed at 0, suggesting the beam won through tactical guard/search rather than explicit fork pressure. Fixed MCTS eval collapsed: Elo 1007, 37.5% vs heuristic over 8 games, with 0W/6D/2L. | Do not scale this combined branch. Decisive tactical-beam replay can fail to transfer into the fixed MCTS agent, especially when the beam's candidate distribution differs from eval-time search. Next work should either align training targets with MCTS policy targets, use teacher/offline data without beam self-play, or evaluate beam-native agents separately as a different product path. |
| 2026-06-04 | `large_16x16_attention_threat_space_draw_pressure` | Kept generated offline two-ply threat-space curriculum and normal MCTS self-play, but raised draw targets to `-0.25` and self-play value loss weight to `0.75`. | Rejected; champion remains `large_16x16_attention_mcts_draw_penalty_final_g12_20260604_082155_525442`. | Stronger draw pressure was correctly applied, but self-play still had 6 full-board draws out of 8 games. Built-in eval was Elo 1039 with 50.0% vs heuristic over 8 games, so no previous-best probe was run. | Stronger value penalty alone does not change MCTS draw behavior. Next local work should change search/action selection in an MCTS-compatible way, such as dynamic temperature, better PUCT/Dirichlet settings, or MCTS target distillation from decisive tactical positions. |
| 2026-06-04 | `large_16x16_attention_threat_space_temp_anneal` | Added MCTS self-play temperature scheduling and configurable Dirichlet noise cutoff. This branch used offline threat-space curriculum, early noise only for about the opening, and sharper late-game temperature while keeping fixed MCTS eval. | Rejected; champion remains `large_16x16_attention_mcts_draw_penalty_final_g12_20260604_082155_525442`. | The mechanism worked: JSONL/TensorBoard log `dirichlet_noise_moves` and `no_noise_moves`, and games showed only 5-19 noisy moves with the rest no-noise. Self-play still drew 5/8 games. Built-in eval was Elo 1039 with 8/8 draws vs heuristic, so no previous-best probe was run. | Simple temperature/noise annealing is not enough. The next MCTS-aligned step should change search targets more directly, for example MCTS policy distillation on curated decisive tactical states, threat-biased root priors, or a stronger terminal/threat evaluator inside search. |
| 2026-06-04 | `large_16x16_attention_threat_prior_mcts` | Added configurable tactical root-prior blending inside MCTS and TensorBoard scalar `self_play/tactical_prior_searches`, while keeping offline threat-space curriculum and capped opening noise. | Rejected; champion remains `large_16x16_attention_mcts_draw_penalty_final_g12_20260604_082155_525442`. | Mechanically worked: self-play logged tactical prior usage on every search move and was 5 decisive games plus 3 draws. Built-in eval was Elo 1023 with 43.75% vs heuristic. A 4-game no-promote probe scored Elo 1031, 37.5% vs heuristic, 50.0% vs previous best, zero failures, and no promotion. TensorBoard events were written under `checkpoints/tensorboard/large_16x16_attention_threat_prior_mcts`. | Soft tactical root priors alone are not enough and may weaken fixed-evaluator strength. Next work should change MCTS targets more directly: policy distillation from curated decisive tactical states, a terminal/threat evaluator inside search, or a stronger supervised teacher before further MCTS scaling. |
| 2026-06-04 | `large_16x16_attention_threat_leaf_mcts` | Added tactical leaf-value backups inside MCTS for immediate wins, multiple immediate losses, and one-ply forcing wins. TensorBoard now logs `self_play/tactical_leaf_evaluations`, positive backups, and negative backups. | Rejected; champion remains `large_16x16_attention_mcts_draw_penalty_final_g12_20260604_082155_525442`. | Mechanically worked but regressed strength and runtime. Self-play was 6 decisive games plus 2 draws with 223 tactical leaf backups, 219 positive and 4 negative. Built-in eval was Elo 991 with 31.25% vs heuristic. A 4-game no-promote probe scored Elo 1045, 50.0% vs heuristic, 50.0% vs previous best, zero failures, and no promotion. TensorBoard events were written under `checkpoints/tensorboard/large_16x16_attention_threat_leaf_mcts`. | Direct tactical leaf values are too blunt as configured and add cost on long games. Do not scale this preset. Next work should either calibrate leaf values with supervised/search distillation, limit them to proven terminal threats only, or generate stronger MCTS policy targets from curated decisive tactical states. |
| 2026-06-04 | `large_16x16_attention_mcts_distill_tactical` | Added curated MCTS policy distillation from hard-position and threat-space boards. The branch logs `mcts_distill/target_mass`, `mcts_distill/target_top_rate`, `mcts_distill/target_entropy`, leaf diagnostics, and `replay/mcts_distill_samples`. | Rejected; champion remains `large_16x16_attention_mcts_draw_penalty_final_g12_20260604_082155_525442`. | The distillation target generator was coherent: 128 positions, split 64 hard-position and 64 threat-space, target mass 0.814, top-hit rate 96.875%, target entropy 0.644. Self-play was 6 decisive games plus 2 draws with 227 forced tactical moves, 500 search moves, and capped opening noise. Built-in eval was Elo 1055 with 56.25% vs heuristic; the 4-game no-promote probe scored Elo 1045, 50.0% vs heuristic, 50.0% vs previous best, zero failures, and no promotion. TensorBoard events were written under `checkpoints/tensorboard/large_16x16_attention_mcts_distill_tactical`. | Short-search distillation on curated tactical states is measurable but not yet strength-improving. Next work should improve target quality before scaling: deeper search on fewer curated positions, distill only decisive PV moves, add calibrated value targets, or compare against supervised threat-space labels without blending weak search priors. |
| 2026-06-04 | `large_16x16_attention_mcts_distill_sharp` | Added stricter MCTS distillation controls: target modes, target-mass/top-hit filtering, accept-rate telemetry, and search-agreement TensorBoard scalars. This preset trains one-hot teacher targets only when 48-playout search already agrees. | Rejected; champion remains `large_16x16_attention_mcts_draw_penalty_final_g12_20260604_082155_525442`. | The filter exposed a useful mismatch: it accepted 96 of 193 candidates, skipped 97, and all accepted samples came from hard-position puzzles. Threat-space candidates did not satisfy the current search-agreement filter. Accepted targets had target mass 1.0, search target mass 0.9996, search top rate 100%, and entropy 0. Self-play was 5 decisive games plus 3 draws with 293 forced tactical moves and 796 search moves. Built-in eval regressed to Elo 991 with 31.25% vs heuristic; a 4-game no-promote probe scored Elo 1045, 50.0% vs heuristic, 50.0% vs previous best, zero failures, and no promotion. | Do not scale strict fork-only distillation. The next branch should preserve the accept-rate/search-agreement telemetry but add source balancing or a threat-space-specific target path, because current MCTS does not agree with the two-ply threat-space generator even when hard-position fork targets look certain. |
| 2026-06-04 | `large_16x16_attention_mcts_distill_balanced_threat` | Added source-aware MCTS distillation quotas, per-source overrides, and per-source TensorBoard scalars. This branch kept hard-position labels strict/search-agreed while accepting threat-space teacher labels despite search disagreement. | Rejected; champion remains `large_16x16_attention_mcts_draw_penalty_final_g12_20260604_082155_525442`. | The telemetry was decisive: 48 hard-position and 48 threat-space samples were trained. Hard-position search target mass was 0.9997 with 100% search top-rate; threat-space search target mass was only 0.0296 with 4.17% search top-rate. Self-play was 4 decisive games plus 4 draws with 342 forced tactical moves and 898 search moves. Built-in eval was Elo 1039 with 50.0% vs heuristic. A 4-game no-promote probe scored Elo 1045, 50.0% vs heuristic, 50.0% vs previous best, zero failures, and no promotion. TensorBoard events were written under `checkpoints/tensorboard/large_16x16_attention_mcts_distill_balanced_threat`. | Do not scale the current threat-space generator as teacher labels. The next useful branch should fix or validate two-ply threat-space semantics against an independent solver/evaluator, or train a separate threat-space classifier/value head before mixing those targets into the main policy. |
| 2026-06-04 | `large_16x16_attention_two_ply_leaf_mcts` | Attempted proof-aware MCTS leaf values for bounded two-ply threats, with reason-level leaf counters. | Aborted locally before final checkpoint; champion remains `large_16x16_attention_mcts_draw_penalty_final_g12_20260604_082155_525442`. | A 10-position independent full-reply diagnostic showed generated threat-space labels are semantically strong: all 10 sampled targets survived all 251 legal replies. However, enabling two-ply proof checks inside every distillation MCTS leaf stalled for several minutes before any distillation event, so the run was stopped as locally impractical. | Do not put bounded two-ply proof checks inside every leaf on the local M5 path. Use cheaper root-only proof hints, cached proof labels, or a separate threat-space auxiliary model instead. |
| 2026-06-04 | `large_16x16_attention_two_ply_prior_mcts` | Added a cheap bounded two-ply root-prior bonus to MCTS plus TensorBoard scalar `self_play/tactical_prior_two_ply_hits`. This branch kept source-aware distillation and avoided per-leaf two-ply proof checks. | Rejected; champion remains `large_16x16_attention_mcts_draw_penalty_final_g12_20260604_082155_525442`. | The root prior solved the distillation visibility problem: threat-space search target mass rose from 0.0296 in the balanced branch to 0.9243, with 93.75% threat-space search top-rate. Self-play was 7 decisive games plus 1 draw, with 400 prior searches and 3 two-ply prior hits. Fixed evaluation still regressed: built-in eval was Elo 1023 with 43.75% vs heuristic, and the 4-game no-promote probe scored Elo 1015, 37.5% vs heuristic, 37.5% vs previous best, zero failures, and no promotion. | Root-only proof hints are cheap and make threat-space labels visible to search, but they do not yet improve fixed play. Next work should avoid adding more tactical priors and instead calibrate value/promotion around these proofs: e.g. separate threat-space value head, proof-labeled evaluation positions, or longer training only after a small probe improves previous-best score. |
| 2026-06-05 | `large_16x16_attention_threat_value_calibration` | Added proof-path value-only training for bounded two-ply threats. Each generated proof contributes root value targets, defender-to-move losing states after the threat, and attacker follow-up states after plausible replies. TensorBoard logs these under `proof_value/*`. | Reclassified after promotion tightening; current local seed remains `large_16x16_attention_mcts_draw_penalty_final_g12_20260604_082155_525442`, Elo 1134. Candidate checkpoint: `large_16x16_attention_threat_value_calibration_final_g20_20260605_073525_079490`. Completion gate remains false. | Proof-value bootstrap generated 750 samples from 128 positions: 125 root states, 125 defender states, 500 follow-up states, and 3 skipped positions. Distillation stayed aligned: overall search target mass 0.9486, threat-space search mass 0.9083, and threat-space top-rate 91.67%. Self-play was 6 decisive games plus 2 draws. Built-in eval was draw-heavy at Elo 1039 with 8/8 draws vs heuristic. Earlier 4-game and 8-game probes showed noisy head-to-head promise, but the stricter re-probe scored Elo 1061, 50.0% vs heuristic, 62.5% vs previous best, zero failures, and failed the new Elo floor by -73 versus the champion. | Proof-path value calibration is useful telemetry but not a remote seed. Promotion now requires at least 16 heuristic games, at least 16 previous-best games, and no fixed-baseline Elo regression. Next work should repeat proof-value calibration only inside a larger 16+ game confidence gate, or use it as auxiliary value data while preserving the 1134-Elo draw-penalty champion as the seed. |
| 2026-06-06 | `large_16x16_top_human_gpu` MVP remote run | Ran the requested remote GPU command from the draw-penalty champion seed with 512 MCTS self-play games, JSONL/TensorBoard runtime instrumentation, GPU inference batching, and parallel self-play workers. After the failed run, added one focused evaluator efficiency rewrite: `eval_parallel_games` plus chunked parallel external evaluation records, and repaired/preserved the eval worker setting in checkpoint metadata. | Rejected; champion remains `large_16x16_attention_mcts_draw_penalty_final_g12_20260604_082155_525442`. Candidate checkpoint: `large_16x16_top_human_gpu_final_g524_20260606_014356_758050`. | Self-play completed 512/512 games with 35,388 moves, zero draws, and zero eval failures. Throughput was about 3.24 self-play moves/sec and 2.80 search moves/sec. GPU batching worked with median about 497.5 positions/batch, but `gpu_smi.log` over the run window had 304 samples with 83.6% idle, 0% median utilization, and 91% max burst; CPU use stayed far below all-core saturation, so self-play was MCTS/search-coordination-bound. Internal eval was Elo 454 with 3.125% vs heuristic; external eval was Elo 684 with 21.875% vs heuristic and 9.375% vs previous best. Replay inspection found balanced decisive value labels but diffuse policy targets: about 84% of sampled targets had max probability <= 0.02. External eval took 5796.9s serially on CPU before the rewrite. | Do not continue this recipe longer. Next remote work should use the parallel evaluator, then address search/target alignment before scaling: sharpen or filter MCTS policy targets, reduce duplicate serial eval, and optimize search/request cadence before trying new heuristics. |

## Things To Investigate

- Whether MCTS helps before the policy/value model is competent.
- Direct policy play plus tactical/threat-search guards as an early training
  stage.
- Local Torch currently reports no MPS backend in this environment; verify the
  remote/local runtime before assuming Apple Silicon or CUDA acceleration.
- Why extra policy/tactical replay reduced or failed to improve head-to-head
  performance; check value targets, draw handling, replay sampling, and
  catastrophic forgetting against the promoted seed.
- Whether defensive tactical examples should be labeled by saved-position
  equity instead of terminal outcome only; the first local probe is
  `large_16x16_attention_balanced_labels`.
- Whether the promoted 8-playout MCTS probe scales when moved to
  `large_16x16_attention_mcts_local`; watch draw rate, self-play value loss,
  and previous-best score before spending remote GPU time.
- Whether lowering the MCTS tactical guard threshold from open-four to
  open-three reduces evaluation draws without causing tactical overforcing.
- Whether a closed-four threshold is the useful middle ground between default
  draw drift and open-three overforcing.
- Whether self-play draws need explicit negative utility instead of neutral
  value labels; the first `-0.10` probe helped enough to promote locally but
  did not remove draw-heavy evaluation.
- Whether lower training loss can be misleading when self-play data is still
  not producing win-converting policy targets; the first scale-up reduced loss
  but lost head-to-head against the champion.
- Whether tactical curriculum needs multi-move hard-position targets rather
  than more single-line win/threat puzzles.
- Whether the self-play/search policy needs an explicit fork-seeking
  threat-search mode; fork targets helped heuristic score but did not improve
  champion head-to-head.
- Whether a candidate that is strong under fork-beam self-play should also be
  evaluated with beam/threat-search instead of only low-playout MCTS.
- Whether forcing-threat replay should be oversampled or assigned stronger
  policy/value targets; the solver branch made self-play decisive but lost
  fixed-evaluator head-to-head, and the first 50% conversion-replay mix
  overfit noisy decisive value targets. A separate policy-only rehearsal step
  was cleaner but still failed fixed evaluation. Explicit high-value labels for
  solver-proven one-move conversions also failed, suggesting the next teacher
  needs deeper multi-move threat-space targets or a stronger evaluator/search
  setup.
- Whether bounded two-ply threat-space should be used only for offline teacher
  data. The first branch improved self-play decisiveness and teacher extraction
  but slowed evaluation and lost to the promoted draw-penalty champion.
- Whether offline two-ply threat-space curriculum should be combined with
  draw-reduction rather than used alone. The first offline branch improved
  heuristic score but still drew too much and failed previous-best promotion.
- Whether tactical-beam self-play targets transfer to fixed MCTS evaluation.
  The first offline threat-space plus fork-beam branch produced 12/12 decisive
  self-play games but collapsed in fixed MCTS eval, so beam replay may need a
  separate distillation target or native beam evaluator.
- Whether stronger draw-value weighting can make MCTS avoid draw lines. The
  first `-0.25` branch did not; search/action selection probably needs to
  change, not only the value target.
- Whether MCTS temperature/noise schedules can reduce draw drift. The first
  annealing branch changed the logged action-selection behavior but still
  evaluated as all draws vs heuristic.
- Whether soft tactical root priors can replace hard tactical forcing. The
  first prior branch increased decisive self-play and logged useful
  `self_play/tactical_prior_searches`, but lost heuristic strength and did not
  beat the champion.
- Whether tactical leaf-value backups can correct the value head inside MCTS.
  The first leaf branch fired often but was nearly all positive backups,
  regressed heuristic score, and made long games slower.
- Whether MCTS-distilled policy targets can align the network with fixed
  evaluator search. The first distillation branch produced coherent target
  scalars but only drew both heuristic and previous-best probes, so target
  quality needs to improve before remote scaling. A stricter distillation
  branch showed that search agreement collapses to hard-position fork targets
  and rejects threat-space targets, which is useful telemetry but not a recipe
  to scale. A source-balanced branch confirmed the deeper problem:
  current MCTS gives generated threat-space targets only about 3% search mass,
  and forcing those labels did not improve fixed evaluation. A proof-aware
  root prior raised threat-space search mass above 92%, but still regressed
  fixed play, so visibility alone is not enough. Proof-path value calibration
  finally promoted a checkpoint locally, but only at small-sample confidence
  and still far below average-human/top-human thresholds.
- Whether forcing-threat replay should be oversampled or assigned stronger
  policy/value targets; the first solver branch made self-play decisive but
  lost fixed-evaluator head-to-head.
- Split-loss telemetry now exists. Watch policy loss and value loss separately
  before changing model/search, especially on self-play updates after tactical
  curriculum.
- Conv-attention, squeeze/excitation, or transformer-style board encoders.
- Hard-position replay and draw-breaking curriculum.
- Learning-rate schedules and value/policy loss weighting.
- Batched inference for search on the remote GPU.
