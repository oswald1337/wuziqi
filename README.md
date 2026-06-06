# Gomoku AlphaZero RL

A Reinforcement Learning agent for Gomoku (Five in a Row) based on the AlphaZero methodology.

## Features
- **AlphaZero Algorithm:** Combines MCTS (Monte Carlo Tree Search) with a deep residual neural network.
- **Multiprocessing:** Uses a Model Server and multiple Self-Play Workers to scale data generation on multi-core CPUs.
- **WandB Integration:** Logs metrics, game replays (HTML), and model checkpoints (Artifacts) to Weights & Biases.
- **Optimized MCTS:** Features virtual loss, GPU batching, and parallel execution.

## Installation

This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync
```

If `uv` is not installed, use the standard Python path:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
```

## Usage

### Training logs and progress

- Raw log tail: `train.log`
- Structured event stream: `checkpoints/training_log.jsonl`
- TensorBoard scalar charts: `checkpoints/tensorboard`
- Structured checkpoint/Elo history: `checkpoints/registry.json`
- Active top-human goal and completion gate: `GOAL.md`
- Local-to-remote recipe and artifact checklist: `REMOTE_GPU_HANDOFF.md`
- Browser dashboard: [http://127.0.0.1:8000](http://127.0.0.1:8000)

The browser dashboard shows the selected checkpoint telemetry, completion
gate checks, promotion checks, total/policy/value loss, replay/bootstrap
state, Elo ladder, and the latest `train.log` plus structured training-event
lines.

For readable scalar charts during long runs:

```bash
.venv/bin/tensorboard --logdir checkpoints/tensorboard --port 6006
```

On the current remote GPU box, TensorBoard is kept on remote port `8081` so it
can be viewed locally at `http://localhost:8080` with:

```bash
ssh -i ~/.ssh/vast -p 59644 root@74.48.140.178 -L 8080:localhost:8081
```

TensorBoard is the default local tracking backend. W&B remains optional for
remote/server runs; `checkpoints/training_log.jsonl` remains the source of
truth and is mirrored into TensorBoard scalars.

For a JSON summary of one run's efficiency and evaluation metrics:

```bash
.venv/bin/python training_audit.py --log checkpoints/training_log.jsonl --preset large_16x16_top_human_gpu --checkpoint-id CHECKPOINT_ID --gpu-log gpu_smi.log --resource-log resource_monitor.jsonl --replay-path checkpoints/replay_16x16_n5_r4_f64.pkl
```

The audit emits a `bottleneck_assessment` block that classifies the run as
`gpu_bound`, `cpu_bound`, `mcts_search_coordination_bound`, or `undetermined`
from measured JSONL/GPU/resource evidence. When `--replay-path` is supplied, it
also emits replay value-label and policy-target quality under `replay_quality`,
plus a `decision_recommendation` that applies the project decision rules.

For replay/data-quality evidence before changing search targets:

```bash
.venv/bin/python replay_audit.py checkpoints/replay_16x16_n5_r4_f64.pkl --max-samples 50000 --strategy tail
```

The replay audit reports value-label balance, draw fraction, policy target
entropy, target sharpness, and a `replay_quality_assessment`.
To compare policy-target sharpening/filtering candidates without changing
training behavior or writing checkpoints, add `--probe-transforms`:

```bash
.venv/bin/python replay_audit.py checkpoints/replay_16x16_n5_r4_f64.pkl --max-samples 50000 --strategy tail --probe-transforms
```

The probe reports each candidate's max-prob/entropy deltas, retained original
MCTS mass before renormalization, support size, top-1 agreement, and distortion
metrics. Use it to shortlist one search-target rewrite before launching another
remote run. Use a smaller `--max-samples` for quick iteration; the 50k replay
tail is the source-of-truth slice and may take a few minutes to load and scan.
The current `large_16x16_top_human_gpu` preset uses the conservative selected
rewrite, `self_play_target_transform=top_k` with `self_play_target_top_k=16`.
Future runs log transform diagnostics such as retained mass, support kept,
top-1 change rate, max-prob delta, and normalized-entropy delta.
New runs also log self-play target-quality scalars directly, such as
`self_play/policy_target_diffuse_fraction`,
`self_play/policy_target_normalized_entropy_mean`, and
`self_play/value_target_draw_fraction`.

To verify the required TensorBoard scalars are present for a preset:

```bash
.venv/bin/python tensorboard_audit.py --logdir checkpoints/tensorboard --preset large_16x16_top_human_gpu
```

For a one-command monitor health check before a long remote run:

```bash
.venv/bin/python remote_health_check.py
```

The health check verifies the tmux monitor sessions, TensorBoard HTTP,
required training/resource TensorBoard scalars, fresh `resource_monitor.jsonl`,
`nvidia-smi`, PyTorch CUDA availability, and the resolved self-play/eval worker
counts for the GPU preset.

For future remote runs, keep a CPU/GPU resource monitor beside TensorBoard:

```bash
.venv/bin/python resource_monitor.py --output resource_monitor.jsonl --interval 30
```

The monitor is cgroup-aware: `cpu_util_percent` is the training container's
CPU usage against its allocated CPU quota, while `host_cpu_util_percent`
captures host-wide pressure that may come from neighboring workloads. The
large GPU preset uses `self_play_parallel_games="auto"` and
`eval_parallel_games="auto"`, which resolve to the usable cgroup CPU worker
count unless the CPU cap is explicitly disabled. The monitor also mirrors
resource scalars to `checkpoints/tensorboard/resource_monitor` by default.

### Shopping baseline

This is the quickest path to a playable local checkpoint. It trains on a small
6x6/four-in-row board, saves checkpoints into `checkpoints/`, and writes
checkpoint metadata plus Elo-style estimates to `checkpoints/registry.json`.

```bash
.venv/bin/python main.py --mode train --preset shopping_baseline
.venv/bin/python main.py --mode web --port 8000
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000), choose a checkpoint,
and play against it. If no trained checkpoint exists yet, the frontend falls
back to baseline agents.

For a faster smoke run:

```bash
.venv/bin/python main.py --mode train --preset debug
```

For a bigger local board:

```bash
.venv/bin/python main.py --mode train --preset bigger_baseline
```

For a 16x16 local board:

```bash
.venv/bin/python main.py --mode train --preset large_16x16
```

Training presets live in [`config.py`](config.py). `bigger_baseline` uses an
8x8 board with five-in-row; `large_16x16` uses a 16x16 board with five-in-row
and intentionally tiny playout/eval settings so it stays responsive locally.

### Iterative improvement loop

Run one or more 16x16 experiments, evaluate each final checkpoint, and record
promotion/gate status in `checkpoints/registry.json`:

```bash
.venv/bin/python main.py --mode improve --presets large_16x16_more_playouts,large_16x16_wider --rounds 3 --eval-games 16 --previous-best-games 16
```

`improve` resumes from the best compatible checkpoint by default, sweeps the
selected hyperparameter presets each round, and stops early if the completion
gate passes. Repeated rounds reuse the architecture-specific replay buffer in
`checkpoints/` and save optimizer state next to trained model checkpoints.

The 16x16 presets also mix in heuristic-vs-random teacher games, tactical
win/block/threat puzzle positions, and anchor distillation from the init
checkpoint. Runtime play has a tactical guard plus a shape-aware prior blended
into MCTS policy inference, which gives the immature network tactical footing
while self-play keeps improving it. Evaluation Elo is recomputed from a fixed
1000 baseline on each probe so repeated evaluations do not inflate ratings.
The improvement loop is not limited to preset sweeping: it is expected to try
training-loop changes, replay/curriculum changes, model architecture changes
such as deeper residual towers or conv-attention, and search alternatives when
MCTS is too expensive or unhelpful for a weak early model.
The local M5/Apple Silicon loop is mainly for discovering the right starting
checkpoint, architecture, curriculum, search strategy, and logging workflow for
the later remote-GPU training run. Local runs should document what works and
what fails so the remote run scales a known recipe instead of repeating tiny
smoke presets. Keep the concrete remote recipe and artifact checklist in
[`REMOTE_GPU_HANDOFF.md`](REMOTE_GPU_HANDOFF.md).
For a single continued run:

```bash
.venv/bin/python main.py --mode train --preset large_16x16_more_playouts --resume-best
```

Use `--fresh` with `--mode improve` only when you intentionally want random
initialization for comparison.

Evaluate a specific checkpoint:

```bash
.venv/bin/python main.py --mode evaluate --agent CHECKPOINT_ID --eval-games 40 --previous-best-games 40
```

For non-promoting diagnostics, evaluate a checkpoint using its native
action-selection mode instead of wrapping it in MCTS:

```bash
.venv/bin/python main.py --mode evaluate --agent CHECKPOINT_ID --eval-games 2 --previous-best-games 0 --eval-mode native --no-promote
```

The current top-human completion gate is defined in [`config.py`](config.py):

- 16x16 board, five-in-row
- at least 90% score against the heuristic baseline over 200 games
- at least 60% score against the previous best checkpoint over 200 games
- estimated local ladder Elo of at least 1800
- zero runtime or illegal-move failures

`config.py` also defines the promotion gate. This is intentionally weaker than
the final top-human gate, but promotion now requires at least 16 heuristic games,
at least 16 previous-best games, and no fixed-baseline Elo regression versus the
current champion. Short smoke loops can still record diagnostics, but they should
not become the default champion. As of the current local registry, the top-human
completion gate is still not passed; keep the improve loop running until the
200-game gate passes.

For longer runs, prefer the explicit long presets rather than repeating tiny
debug loops. These are starting points, not a claim that the listed game or
playout counts are sufficient for strong play:

```bash
.venv/bin/python main.py --mode improve --presets large_16x16_attention_smoke --rounds 1 --eval-games 4 --previous-best-games 4
.venv/bin/python main.py --mode train --preset large_16x16_attention_policy_local --resume-best
.venv/bin/python main.py --mode train --preset large_16x16_attention_beam_smoke --resume-best
.venv/bin/python main.py --mode train --preset large_16x16_attention_value_light --resume-best
.venv/bin/python main.py --mode train --preset large_16x16_attention_delayed_value --resume-best
.venv/bin/python main.py --mode train --preset large_16x16_attention_balanced_labels --resume-best
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_probe --resume-best
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_local --resume-best
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_aggressive --resume-best
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_closed_four --resume-best
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_draw_penalty --resume-best
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_draw_penalty_scale --resume-best
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_win_conversion --resume-best
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_fork_teacher --resume-best
.venv/bin/python main.py --mode train --preset large_16x16_attention_fork_beam --resume-best
.venv/bin/python main.py --mode train --preset large_16x16_attention_threat_solver --resume-best
.venv/bin/python main.py --mode train --preset large_16x16_attention_conversion_replay --resume-best
.venv/bin/python main.py --mode train --preset large_16x16_attention_conversion_policy_replay --resume-best
.venv/bin/python main.py --mode train --preset large_16x16_attention_conversion_teacher --resume-best
.venv/bin/python main.py --mode train --preset large_16x16_attention_threat_space_teacher --resume-best
.venv/bin/python main.py --mode train --preset large_16x16_attention_threat_space_offline --resume-best
.venv/bin/python main.py --mode train --preset large_16x16_attention_threat_space_fork_beam --resume-best
.venv/bin/python main.py --mode train --preset large_16x16_attention_threat_space_draw_pressure --resume-best
.venv/bin/python main.py --mode train --preset large_16x16_attention_threat_space_temp_anneal --resume-best
.venv/bin/python main.py --mode train --preset large_16x16_attention_threat_prior_mcts --resume-best
.venv/bin/python main.py --mode train --preset large_16x16_attention_threat_leaf_mcts --resume-best
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_distill_tactical --resume-best
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_distill_sharp --resume-best
.venv/bin/python main.py --mode train --preset large_16x16_attention_mcts_distill_balanced_threat --resume-best
.venv/bin/python main.py --mode train --preset large_16x16_attention_two_ply_prior_mcts --resume-best
.venv/bin/python main.py --mode train --preset large_16x16_attention_threat_value_calibration --resume-best
.venv/bin/python main.py --mode improve --presets large_16x16_long_cpu --rounds 2 --eval-games 16 --previous-best-games 16
.venv/bin/python main.py --mode improve --presets large_16x16_long_mps --rounds 3 --eval-games 24 --previous-best-games 24
.venv/bin/python main.py --mode improve --presets large_16x16_attention_mps --rounds 3 --eval-games 24 --previous-best-games 24
.venv/bin/python main.py --mode improve --presets large_16x16_top_human_gpu --rounds 10 --eval-games 32 --previous-best-games 32
```

The 4-game smoke command above is intentionally below the promotion-game floor.
Use it only for diagnostics.

`large_16x16_attention_smoke` uses direct policy self-play with tactical
guards. It is meant to test whether the conv-attention model can absorb
tactical curriculum cheaply before spending local MPS or remote GPU time on
deeper MCTS.
`large_16x16_attention_policy_local` keeps that cheaper action-selection path
but gives it a larger curriculum and 16 self-play games, which is the next
local step when repeated smoke continuations stop improving.
`large_16x16_attention_beam_smoke` is the next staged-search probe: it keeps
the tactical guard, then evaluates a small beam of policy/tactical candidates
with one-ply network value before sampling a training target.
`large_16x16_attention_value_light` downweights value loss after split-loss
telemetry showed self-play value loss rising much faster than supervised
tactical value loss.
`large_16x16_attention_delayed_value` goes one step further: curriculum and
bootstrap data still train the value head, but self-play updates are policy
only until the value targets are less suspect.
`large_16x16_attention_balanced_labels` keeps the same cheap policy/tactical
loop but raises defensive block puzzle value targets, so immediate blocks and
threat blocks stop looking like near-losing moves to the value head.
`large_16x16_attention_mcts_probe` is a tiny low-playout MCTS stage using the
same conv-attention model and balanced tactical labels. It is meant to measure
whether search creates better replay before scaling MCTS locally or remotely.
`large_16x16_attention_mcts_local` is the modest follow-up: 16 self-play games
with 16 playout search, larger replay, and anchor distillation. Use it after a
tiny MCTS probe promotes.
`large_16x16_attention_mcts_aggressive` keeps 16-playout MCTS but lowers the
tactical guard threshold to open-three strength. It is a local anti-draw probe
for testing whether the agent needs more forced threat play before scaling
playouts.
`large_16x16_attention_mcts_closed_four` is the middle threshold probe: it
forces closed-four-or-better threats, avoiding the overforcing seen when every
open three is forced.
`large_16x16_attention_mcts_draw_penalty` keeps the open-three guard but
relabels drawn self-play positions to a small negative value, testing whether
the value head needs an explicit anti-draw utility before larger MCTS runs.
`large_16x16_attention_mcts_draw_penalty_scale` is the cautious local scale-up
of that recipe: 16 self-play games, 24-playout search, larger tactical
curriculum, and a larger replay buffer.
`large_16x16_attention_mcts_win_conversion` keeps the current champion recipe
small but over-samples attacking tactical puzzles to test whether the model
needs better win-conversion targets before any more MCTS scaling.
`large_16x16_attention_mcts_fork_teacher` adds hard-position puzzles where the
target move creates two simultaneous open-four threats, testing whether
multi-move win-conversion data is the missing ingredient.
`large_16x16_attention_fork_beam` keeps those hard-position puzzles but swaps
self-play to a fork-weighted tactical beam, so local runs can test whether
cheap threat-search creates better replay before scaling MCTS again.
`large_16x16_attention_threat_solver` returns to MCTS but adds a cheap
forcing-win solver for moves that create multiple immediate winning replies.
It is meant to separate useful draw-breaking tactical conversion from noisy
open-three overforcing.
`large_16x16_attention_conversion_replay` keeps that solver and adds a recent
decisive replay lane mixed into self-play training batches, so conversion
examples do not vanish inside the large uniform replay buffer.
`large_16x16_attention_conversion_policy_replay` keeps normal self-play value
training but adds a separate policy-only rehearsal step for recent decisive
conversion positions, avoiding the noisy value labels that hurt the first
conversion-replay branch.
`large_16x16_attention_conversion_teacher` reconstructs self-play positions
where the threat solver proves a forcing conversion and relabels them with
explicit high-value teacher targets.
`large_16x16_attention_threat_space_teacher` extends that idea with a bounded
two-ply threat-space teacher and optional two-ply tactical forcing in MCTS. The
first local probe made self-play decisive but did not beat the fixed champion,
so treat it as a diagnostic branch rather than a remote-GPU seed.
`large_16x16_attention_threat_space_offline` keeps runtime MCTS cheap and uses
generated two-ply threat-space positions as bootstrap curriculum. Its first
probe improved heuristic score but still did not beat the promoted champion.
`large_16x16_attention_threat_space_fork_beam` combines that offline
curriculum with fork-weighted tactical-beam self-play. It produced decisive
self-play but failed fixed MCTS evaluation, so it is diagnostic only.
`large_16x16_attention_threat_space_draw_pressure` keeps MCTS self-play but
raises draw penalty and self-play value weight. Its first probe did not reduce
MCTS draw drift, so it is diagnostic only.
`large_16x16_attention_threat_space_temp_anneal` keeps MCTS self-play but
uses early Dirichlet noise and sharper late-game temperature. Its first probe
still evaluated as draw-heavy, so it is diagnostic only.
`large_16x16_attention_threat_prior_mcts` keeps offline threat-space
curriculum and blends a tactical shape prior into the MCTS root prior instead
of hard-forcing every lower-threshold shape. Its first probe made self-play
more decisive but failed heuristic and previous-best evaluation, so do not
scale soft root priors alone.
`large_16x16_attention_threat_leaf_mcts` adds tactical value backups at MCTS
leaves for immediate wins, unavoidable losses, and one-ply forcing wins. It
worked mechanically and logged `self_play/tactical_leaf_*` scalars, but its
first probe was slower and weaker than the champion, so treat it as diagnostic
unless the leaf values are recalibrated or narrowed.
`large_16x16_attention_mcts_distill_tactical` builds curated hard-position and
threat-space boards, converts short MCTS searches into policy targets, and logs
`mcts_distill/*` plus `replay/mcts_distill_samples` scalars. Its first local
probe produced coherent targets but did not beat the champion, so it is a
diagnostic starting point for better search-target generation rather than a
remote seed recipe.
`large_16x16_attention_mcts_distill_sharp` filters MCTS distillation to
teacher targets that the search already agrees with, then trains one-hot policy
targets and logs `mcts_distill/accept_rate` plus raw search agreement. Its
first local probe accepted only hard-position targets, rejected the
threat-space half, and regressed fixed evaluation, so do not scale strict
fork-only distillation directly.
`large_16x16_attention_mcts_distill_balanced_threat` adds source-aware
distillation quotas and per-source TensorBoard stats. It preserves
threat-space teacher labels even when MCTS disagrees, while keeping
hard-position targets search-agreed. Its first probe proved the mismatch
clearly: hard-position search mass was near 1.0, threat-space search mass was
near 0.03, and fixed evaluation still did not improve. Use this telemetry for
target-generator work; do not scale the current threat-space labels as-is.
`large_16x16_attention_two_ply_prior_mcts` keeps the source-aware
distillation setup but adds a cheap bounded two-ply root-prior bonus to MCTS.
It made threat-space targets visible to distillation search and kept self-play
mostly decisive, but fixed evaluation still regressed, so treat this as useful
search telemetry rather than a remote seed.
`large_16x16_attention_threat_value_calibration` adds value-only proof-path
training for bounded two-ply threats: the root state is favorable, the
defender-to-move state after the threat is unfavorable, and attacker follow-up
states are favorable. Its first local run promoted as the current 16x16
conv-attention checkpoint, but it is still far below the top-human gate.

### Full training

To start training the agent:

```bash
uv run python main.py --mode train --preset full
```

**Arguments:**
- `--debug`: Run in debug mode with fewer workers and simulations for testing.
- `--preset`: Choose `debug`, `shopping_baseline`, `bigger_baseline`, `large_16x16`, `large_16x16_more_playouts`, `large_16x16_tactical`, `large_16x16_tactical_deep`, `large_16x16_curriculum`, `large_16x16_attention_smoke`, `large_16x16_attention_policy_local`, `large_16x16_attention_beam_smoke`, `large_16x16_attention_value_light`, `large_16x16_attention_delayed_value`, `large_16x16_attention_balanced_labels`, `large_16x16_attention_mcts_probe`, `large_16x16_attention_mcts_local`, `large_16x16_attention_mcts_aggressive`, `large_16x16_attention_mcts_closed_four`, `large_16x16_attention_mcts_draw_penalty`, `large_16x16_attention_mcts_draw_penalty_scale`, `large_16x16_attention_mcts_win_conversion`, `large_16x16_attention_mcts_fork_teacher`, `large_16x16_attention_fork_beam`, `large_16x16_attention_threat_solver`, `large_16x16_attention_conversion_replay`, `large_16x16_attention_conversion_policy_replay`, `large_16x16_attention_conversion_teacher`, `large_16x16_attention_threat_space_teacher`, `large_16x16_attention_threat_space_offline`, `large_16x16_attention_threat_space_fork_beam`, `large_16x16_attention_threat_space_draw_pressure`, `large_16x16_attention_threat_space_temp_anneal`, `large_16x16_attention_threat_prior_mcts`, `large_16x16_attention_threat_leaf_mcts`, `large_16x16_attention_mcts_distill_tactical`, `large_16x16_attention_mcts_distill_sharp`, `large_16x16_attention_mcts_distill_balanced_threat`, `large_16x16_attention_two_ply_leaf_mcts`, `large_16x16_attention_two_ply_prior_mcts`, `large_16x16_attention_threat_value_calibration`, `large_16x16_long_cpu`, `large_16x16_long_mps`, `large_16x16_attention_mps`, `large_16x16_top_human_gpu`, `large_16x16_wider`, or `full`.

### WandB Authentication (Remote Servers)

If you are running this on a remote server (especially one you don't fully trust), **do not run `wandb login`**, as it saves credentials to a file on disk.

Instead, use an environment variable to authenticate for the single session:

1.  Get your API key from [https://wandb.ai/authorize](https://wandb.ai/authorize).
2.  Run the training command with the key:

```bash
WANDB_API_KEY=your_api_key_here uv run python main.py --mode train
```

This keeps your credentials in memory only for that process.

## Architecture

- **`main.py`**: Entry point.
- **`config.py`**: Board sizes, network sizes, playout counts, and training presets.
- **`GOAL.md`**: Active top-human objective, gates, and local-to-remote training policy.
- **`REMOTE_GPU_HANDOFF.md`**: Remote GPU recipe, artifact checklist, and experiment notes.
- **`train.py`**: Training pipeline, Model Server, and Worker logic.
- **`mcts.py`**: Monte Carlo Tree Search implementation.
- **`model.py`**: PyTorch neural network definition.
- **`game.py`**: Gomoku game rules and board logic.
- **`checkpoint_registry.py`**: Checkpoint metadata and Elo-style registry.
- **`evaluator.py`**: Fixed tournament evaluator, promotion logic, and completion gate checks.
- **`improve.py`**: Iterative train/evaluate loop over experiment presets.
- **`players.py`**: Random and heuristic baseline players.
- **`tactical.py`**: Shared tactical scoring for wins, blocks, threats, and move priors.
- **`web_app.py`** / **`static/`**: Local API and browser UI for playing against checkpoints.
