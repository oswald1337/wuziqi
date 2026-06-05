# Top-Human Wuziqi Agent Goal

Train the 16x16 five-in-row AlphaZero agent until it approaches top human
playing strength, with enough logs and frontend telemetry to watch long runs
without guessing what the trainer is doing.

The previous average-human target is now only a milestone. The active goal is
top-human performance.

## Completion Gate

This goal remains active until a promoted checkpoint satisfies all top-human
completion checks:

- Board: 16x16, five-in-row.
- Heuristic baseline: score at least 90% over 200 alternating-start games.
- Previous best checkpoint: score at least 60% over 200 alternating-start games.
- Elo: reach at least 1800 from the fixed-baseline local evaluator.
- Reliability: zero illegal-move or runtime failures during gate evaluation.
- Experiment tracking: serious long runs mirror JSONL events into TensorBoard
  scalars under `checkpoints/tensorboard` so progress can be monitored live.
- Human/top-human proxy: when enough data exists, score at least 55% over at
  least 50 logged games against the project owner or a stronger external proxy.

## Milestones

1. Working model: legal 16x16 play, frontend checkpoint selector, visible logs.
2. Average-human milestone: 70% vs heuristic over 40 games, 55% vs previous
   best over 40 games, Elo at least 1200.
3. Strong club milestone: 80% vs heuristic over 100 games, 58% vs previous
   best over 100 games, Elo at least 1500.
4. Top-human completion: pass the completion gate above.

## Long-Run Training Policy

Short 4-game smoke loops are only for debugging. Serious improvement runs
should train long enough to make replay and optimizer persistence matter. The
named presets below are starting points, not a claim that 48 or 96 playouts, a
few dozen games, or the current AlphaZero loop are enough for strong play.

Local training on the M5 Apple Silicon machine is primarily a discovery and
preparation loop. It should find a promising starting checkpoint, training
recipe, architecture, curriculum, search setup, and logging workflow for the
real heavy training run, which is expected to happen later on a remote server
with a stronger GPU. Local progress still matters, but the local loop should
not pretend that small self-play counts or low-playout MCTS are enough for the
final playing-strength target.

The improvement loop is allowed, and expected, to change more than numeric
hyperparameters. To reach a good Gomoku agent, the loop may redesign and test:

- the training loop itself: replay sampling, curriculum order, optimizer,
  learning-rate schedules, value/policy loss weighting, batching, augmentation,
  evaluation gates, promotion logic, and checkpoint selection
- the model architecture: deeper or wider residual towers, bottleneck blocks,
  policy/value heads, attention or conv-attention blocks, squeeze/excitation,
  transformer-style board encoders, or other architectures that fit 16x16 play
- the search and action-selection stack: MCTS playout count, PUCT constants,
  temperature schedules, tactical priors, direct policy play, beam search,
  alpha-beta/threat search hybrids, or reducing MCTS dependence until the model
  is strong enough for MCTS to be useful
- the data engine: more self-play, stronger tactical generators, opening
  diversity, hard-position replay, teacher games, human games, and external
  engine/proxy games when available
- the hardware path: CPU, Apple Silicon MPS on the local M5 chip, CUDA where
  available, or mixed CPU/search plus accelerator/batched-inference setups

Every serious experiment should be documented in `checkpoints/registry.json`,
`checkpoints/training_log.jsonl`, and, when it changes the intended direction,
this goal document, README, or `REMOTE_GPU_HANDOFF.md`.

## Local-To-Remote Handoff

The local M5 loop should prepare the remote GPU run by producing:

- a best current checkpoint to seed remote training
- the most promising architecture candidate so far
- a documented training recipe: preset, model size, curriculum, search method,
  replay policy, optimizer, learning-rate schedule, and evaluation gate
- evidence about which ideas failed or regressed, especially around MCTS cost,
  draw-heavy training, curriculum-only training, and tactical forgetting
- clean logs and dashboard views that can be reused on the remote server
- a remote run command or config that scales the local recipe instead of
  blindly repeating small smoke presets

Maintain the concrete remote recipe in `REMOTE_GPU_HANDOFF.md`.

If local experiments show that MCTS is not useful before the model has learned
basic policy/value structure, the loop should try cheaper alternatives first,
for example direct policy play with tactical guards, threat-search hybrids,
supervised/tactical pretraining, or staged training where MCTS is increased only
after the model crosses a measurable baseline.

Use these presets from `config.py`:

- `large_16x16_tactical`: quick tactical smoke run.
- `large_16x16_tactical_deep`: medium tactical run.
- `large_16x16_long_cpu`: longer local CPU run with 256 self-play games and
  96-playout search.
- `large_16x16_long_mps`: longer local Apple Silicon run for the M5/MPS path,
  with 384 self-play games and 128-playout search.
- `large_16x16_attention_smoke`: quick conv-attention discovery run using
  direct policy self-play with tactical guards, for testing pre-MCTS learning.
- `large_16x16_attention_policy_local`: longer local CPU-safe conv-attention
  policy/tactical run with more curriculum and 16 cheap self-play games.
- `large_16x16_attention_beam_smoke`: quick conv-attention staged-search run
  using tactical guard plus one-ply policy/value beam search.
- `large_16x16_attention_value_light`: quick conv-attention policy/tactical
  run with value loss downweighted after split-loss telemetry showed noisy
  self-play value targets.
- `large_16x16_attention_delayed_value`: quick conv-attention policy/tactical
  run that trains value on curriculum/bootstrap data but disables value loss
  on self-play updates.
- `large_16x16_attention_balanced_labels`: quick conv-attention policy/tactical
  run with configurable tactical value labels and higher defensive block
  targets, for testing whether value learning is being poisoned by low block
  labels.
- `large_16x16_attention_mcts_probe`: tiny low-playout conv-attention MCTS
  probe using balanced tactical labels and anchor distillation, for testing
  whether search replay is worth scaling.
- `large_16x16_attention_mcts_local`: modest local follow-up after the tiny
  MCTS probe promotes, using 16 self-play games, 16 playouts, larger replay,
  and anchor distillation.
- `large_16x16_attention_mcts_aggressive`: local anti-draw MCTS probe that
  keeps 16 playouts but lowers the tactical guard threshold to open-three
  strength, so the search stack forces more threat play.
- `large_16x16_attention_mcts_closed_four`: middle tactical-threshold MCTS
  probe that forces closed-four-or-better threats while avoiding open-three
  overforcing.
- `large_16x16_attention_mcts_draw_penalty`: anti-draw training-loop probe
  that keeps the open-three tactical guard but labels drawn self-play positions
  with a small negative value.
- `large_16x16_attention_mcts_draw_penalty_scale`: cautious local scale-up of
  the draw-penalty recipe, with 16 self-play games, 24-playout search, larger
  replay, stronger curriculum, and more anchor distillation.
- `large_16x16_attention_mcts_win_conversion`: attacking tactical-curriculum
  branch that over-samples win/threat conversion targets before adding more
  MCTS cost.
- `large_16x16_attention_mcts_fork_teacher`: hard-position branch that teaches
  moves creating two simultaneous open-four threats, testing multi-move
  win-conversion data before further MCTS scaling.
- `large_16x16_attention_fork_beam`: fork-weighted tactical beam branch that
  tries to make cheap self-play actively create those double-threat positions.
- `large_16x16_attention_threat_solver`: MCTS branch with a cheap forcing-win
  solver that detects moves creating multiple immediate winning replies, for
  testing whether stricter draw-breaking tactics beat noisy open-three forcing.
- `large_16x16_attention_conversion_replay`: threat-solver MCTS branch with a
  recent decisive replay lane, testing whether conversion examples survive
  uniform replay sampling when mixed into self-play updates.
- `large_16x16_attention_conversion_policy_replay`: safer conversion-replay
  branch that keeps normal self-play value training but rehearses recent
  decisive conversion positions with a separate policy-only step.
- `large_16x16_attention_conversion_teacher`: branch that reconstructs
  self-play positions where the threat solver proves a forcing conversion and
  relabels those moves with explicit high-value teacher targets.
- `large_16x16_attention_threat_prior_mcts`: MCTS branch that keeps offline
  threat-space curriculum and blends a tactical shape prior into root search,
  testing soft search guidance after hard forcing and annealing failed.
- `large_16x16_attention_threat_leaf_mcts`: MCTS branch that backs up
  tactical leaf values for immediate wins, unavoidable losses, and one-ply
  forcing wins, testing value-side search correction.
- `large_16x16_attention_mcts_distill_tactical`: MCTS-aligned distillation
  branch that builds curated hard-position and threat-space boards, turns a
  short MCTS search into policy targets, and tracks whether those targets
  transfer into fixed-evaluator strength.
- `large_16x16_attention_mcts_distill_sharp`: stricter MCTS-distillation
  branch that filters for search-agreed teacher targets and logs accept-rate
  and raw-search agreement, testing whether sharper policy targets help.
- `large_16x16_attention_mcts_distill_balanced_threat`: source-aware
  distillation branch that keeps hard-position targets strict while preserving
  threat-space teacher targets and logging per-source search agreement.
- `large_16x16_attention_two_ply_prior_mcts`: proof-aware root-prior branch
  that biases MCTS toward bounded two-ply threats once at the root, avoiding
  the much higher cost of checking two-ply threats at every leaf.
- `large_16x16_attention_threat_value_calibration`: value-calibration branch
  that trains proof-path root, defender-to-move, and follow-up states around
  bounded two-ply threats before source-aware MCTS distillation.
- `large_16x16_attention_mps`: local Apple Silicon conv-attention experiment,
  with 256 self-play games and 96-playout search.
- `large_16x16_top_human_gpu`: serious GPU run with 512 self-play games and
  256-playout search using the conv-attention architecture.

Long runs should:

1. Resume from the best compatible checkpoint unless deliberately testing a
   fresh baseline.
2. Preserve the init checkpoint with anchor distillation before adding new
   tactical curriculum.
3. Mix tactical teacher games, win/block/threat puzzles, and self-play.
4. Save mid-run checkpoints frequently enough to inspect progress, but not so
   often that checkpoint churn dominates training.
5. Persist replay buffers and optimizer state in `checkpoints/`.
6. Evaluate quick candidates with 16 to 32 games, then only run 200-game gates
   on candidates that look genuinely stronger.
7. Recompute Elo from a fixed 1000 baseline every evaluation so repeated probes
   cannot inflate a stale champion.
8. Promote checkpoints only when they meet the promotion-game floor and do not
   regress fixed-baseline Elo versus the current champion; short smoke probes are
   diagnostics, not champion evidence.

## Progress Visibility

The frontend and log files must make long runs watchable:

- Raw trainer log: `train.log`
- Structured event stream: `checkpoints/training_log.jsonl`
- TensorBoard scalar charts: `checkpoints/tensorboard`
- Checkpoint registry, Elo, promotion, and gate records:
  `checkpoints/registry.json`
- Training presets and gates: `config.py`
- Browser dashboard: `http://127.0.0.1:8000`

The dashboard should continue to show:

- selected checkpoint and Elo
- current completion gate status
- promotion status
- total, policy, and value loss plus entropy
- replay sample count
- expert, puzzle, anchor, tactical-forcing, and self-play activity
- recent self-play games
- evaluation rows against random, heuristic, and previous best
- log tail from `train.log` and `checkpoints/training_log.jsonl`

## Experiment Tracking

Readable scalar tracking is required for all serious runs.

TensorBoard is the default local tracking backend, while W&B remains optional
for remote/server runs. JSONL remains the source-of-truth event log, but it
must also be mirrored into readable charts.

TensorBoard should log at least:

- `train/loss`
- `train/policy_loss`
- `train/value_loss`
- `train/entropy`
- `self_play/moves`
- `self_play/draw_rate`
- `self_play/forced_tactical_moves`
- `self_play/search_moves`
- `eval/elo`
- `eval/heuristic_score`
- `eval/previous_best_score`
- `replay/samples`

Experimental branches may add extra scalars, for example
`train/priority_samples`, `conversion/loss`,
`conversion/policy_loss`, `conversion/value_loss`, and
`replay/conversion_samples`, plus conversion-teacher scalars such as
`conversion_teacher/loss`, `conversion_teacher/policy_loss`, and
`replay/conversion_teacher_samples`, and threat-space diagnostics such as
`self_play/two_ply_threat_moves`, `self_play/tactical_prior_searches`, or
`self_play/tactical_leaf_evaluations`. MCTS-distillation branches should also
log `mcts_distill/target_mass`, `mcts_distill/target_top_rate`,
`mcts_distill/target_entropy`, `mcts_distill/search_target_mass`,
`mcts_distill/search_top_rate`, `mcts_distill/accept_rate`,
`mcts_distill/skipped`, `mcts_distill/leaf_evaluations`, and
`replay/mcts_distill_samples`. Source-aware branches should additionally log
per-source tags such as `mcts_distill/hard_position/search_target_mass` and
`mcts_distill/threat_space/search_target_mass`. Proof-aware prior branches
should also log `self_play/tactical_prior_two_ply_hits`.

Completion requires that long-running experiments can be monitored with:

```bash
.venv/bin/tensorboard --logdir checkpoints/tensorboard --port 6006
```

## Useful Commands

Quick smoke loop:

```bash
.venv/bin/python main.py --mode improve --presets large_16x16_tactical,large_16x16_tactical_deep --rounds 3 --eval-games 8 --previous-best-games 8
```

Conv-attention policy/tactical smoke loop:

```bash
.venv/bin/python main.py --mode improve --presets large_16x16_attention_smoke --rounds 1 --eval-games 4 --previous-best-games 4
```

Longer local conv-attention policy/tactical run:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_policy_local --resume-best
```

Conv-attention tactical beam smoke run:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_beam_smoke --resume-best
```

Conv-attention value-light smoke run:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_value_light --resume-best
```

Conv-attention delayed-value smoke run:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_delayed_value --resume-best
```

Conv-attention balanced tactical-label smoke run:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_balanced_labels --resume-best
```

Conv-attention low-playout MCTS probe:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_probe --resume-best
```

Modest local conv-attention MCTS scale-up:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_local --resume-best
```

Aggressive tactical-guard MCTS anti-draw probe:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_aggressive --resume-best
```

Closed-four tactical-guard MCTS threshold probe:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_closed_four --resume-best
```

Draw-penalty tactical MCTS probe:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_draw_penalty --resume-best
```

Draw-penalty local scale-up:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_draw_penalty_scale --resume-best
```

Win-conversion tactical curriculum probe:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_win_conversion --resume-best
```

Fork-teacher hard-position probe:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_fork_teacher --resume-best
```

Fork-weighted beam self-play probe:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_fork_beam --resume-best
```

Forcing-threat solver MCTS probe:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_threat_solver --resume-best
```

Conversion-replay MCTS probe:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_conversion_replay --resume-best
```

Policy-only conversion-replay probe:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_conversion_policy_replay --resume-best
```

Conversion-teacher probe:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_conversion_teacher --resume-best
```

Bounded two-ply threat-space teacher probe:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_threat_space_teacher --resume-best
```

Offline two-ply threat-space curriculum probe:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_threat_space_offline --resume-best
```

Offline threat-space plus fork-beam self-play probe:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_threat_space_fork_beam --resume-best
```

Offline threat-space plus stronger MCTS draw-pressure probe:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_threat_space_draw_pressure --resume-best
```

Offline threat-space plus MCTS temperature/noise annealing probe:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_threat_space_temp_anneal --resume-best
```

Offline threat-space plus tactical root-prior MCTS probe:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_threat_prior_mcts --resume-best
```

Offline threat-space plus tactical leaf-value MCTS probe:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_threat_leaf_mcts --resume-best
```

Curated tactical MCTS policy-distillation probe:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_distill_tactical --resume-best
```

Strict accepted-target MCTS policy-distillation probe:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_distill_sharp --resume-best
```

Source-balanced threat-space MCTS policy-distillation probe:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_distill_balanced_threat --resume-best
```

Proof-aware two-ply root-prior MCTS probe:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_two_ply_prior_mcts --resume-best
```

Threat-space proof-path value-calibration probe:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_attention_threat_value_calibration --resume-best
```

Long local CPU run:

```bash
.venv/bin/python main.py --mode improve --presets large_16x16_long_cpu --rounds 2 --eval-games 16 --previous-best-games 16
```

Long local Apple Silicon run:

```bash
.venv/bin/python main.py --mode improve --presets large_16x16_long_mps --rounds 3 --eval-games 24 --previous-best-games 24
```

Local conv-attention experiment:

```bash
.venv/bin/python main.py --mode improve --presets large_16x16_attention_mps --rounds 3 --eval-games 24 --previous-best-games 24
```

Serious GPU/top-human run:

```bash
.venv/bin/python main.py --mode improve --presets large_16x16_top_human_gpu --rounds 10 --eval-games 32 --previous-best-games 32
```

Top-human gate evaluation:

```bash
.venv/bin/python main.py --mode evaluate --agent CHECKPOINT_ID --eval-games 200 --previous-best-games 200
```

Frontend:

```bash
.venv/bin/python main.py --mode web --port 8000
```
