# Remote GPU MVP Wuziqi Agent Goal

Train the 16x16 five-in-row AlphaZero/Gomoku agent toward top-human strength,
but move the project out of local micro-trial mode and into a smaller number of
longer, instrumented GPU trials.

The immediate objective is an MVP remote training loop: a real checkpoint that
trains for long enough to expose convergence behavior, runtime bottlenecks, and
search/data weaknesses, with logs clear enough to decide the next rewrite from
evidence.

## Completion Gate

The top-human goal remains active until a promoted checkpoint satisfies all
completion checks:

- Board: 16x16, five-in-row.
- Heuristic baseline: score at least 90% over 200 alternating-start games.
- Previous best checkpoint: score at least 60% over 200 alternating-start games.
- Elo: reach at least 1800 from the fixed-baseline evaluator.
- Reliability: zero illegal-move or runtime failures during gate evaluation.
- Experiment tracking: long runs mirror JSONL events into TensorBoard scalars
  under `checkpoints/tensorboard`.
- Human/top-human proxy: when enough data exists, score at least 55% over at
  least 50 logged games against the project owner or a stronger external proxy.

Average-human and strong-club strength are milestones, not completion.

## Current Seed

Start the remote stage from the strongest current local seed:

`large_16x16_attention_mcts_draw_penalty_final_g12_20260604_082155_525442`

Known local evidence:

- Elo: 1134.
- 16-game heuristic score: 56.25%.
- 16-game previous-best score: 59.375%.
- Zero evaluation failures.
- Still far below the final gate.

The proof-value calibration branch is useful telemetry, but it is not the seed:
its fixed-baseline re-probe regressed below this champion.

## Remote MVP Policy

Stop running many tiny local branches. The next stage should use fewer, longer,
more diagnostic runs:

1. Run one remote GPU MVP trial from the current seed.
2. Watch TensorBoard and JSONL logs during the run.
3. Measure where time goes: self-play search, neural inference, training,
   tactical generators, evaluation, checkpoint writes, and logging.
4. Inspect convergence: policy loss, value loss, entropy, draw rate, search
   moves, forced tactical moves, replay size, Elo, and evaluation scores.
5. Make at most one focused rewrite before the next long trial unless the run is
   clearly broken.

Preferred first trial:

```bash
.venv/bin/python main.py --mode improve --presets large_16x16_top_human_gpu --rounds 1 --eval-games 32 --previous-best-games 32
```

If the first GPU run is too slow or bottlenecked by CPU-side MCTS, do not start
five more variants. Profile first, then change the bottleneck.

## What May Be Changed

The remote MVP loop may rewrite or simplify parts of the system when logs show a
real bottleneck or convergence failure:

- Training loop: batching, replay sampling, optimizer persistence, learning-rate
  schedule, loss weighting, checkpoint cadence, evaluation cadence.
- Search: MCTS playouts, direct policy play, tactical guards, threat-search
  hybrids, batched inference, or staged search that increases cost only after
  the policy becomes useful.
- Data: self-play volume, tactical teacher data, hard-position replay, opening
  diversity, data augmentation, and replay deduplication.
- Model: conv-attention size, residual depth/width, heads, or attention blocks,
  but only when logs suggest model capacity is the limiting factor.
- Hardware path: CUDA device use, CPU/GPU split, multiprocessing, and remote
  monitoring.

Data augmentation is allowed and encouraged when it is cheap and semantically
safe for 16x16 Gomoku, especially rotations/reflections already used by
self-play. Any new augmentation must be covered by tests.

## Efficiency Requirements

Every serious remote run must record enough evidence to answer:

- Are we GPU-bound, CPU-bound, or MCTS-bound?
- How many self-play moves per second are we getting?
- How much time is spent in search versus neural training?
- Are losses decreasing or just oscillating?
- Is value loss learning anything useful, or is it noisy draw/value churn?
- Is entropy falling too fast, staying flat, or collapsing?
- Are draw rates improving?
- Does Elo improve against fixed baselines without promotion noise?

If these questions cannot be answered from existing logs, improve logging before
starting another long run.

## Required Tracking

JSONL remains the source-of-truth event log:

`checkpoints/training_log.jsonl`

TensorBoard is the default readable chart backend:

```bash
.venv/bin/tensorboard --logdir checkpoints/tensorboard --host 0.0.0.0 --port 8080
```

The local tunnel for viewing TensorBoard is:

```bash
TERM=xterm-256color ssh -i ~/.ssh/vast -p 59644 root@74.48.140.178 -L 8080:localhost:8080
```

TensorBoard should show at least:

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

Useful extra scalars include runtime timings, moves per second, search seconds,
train seconds, evaluation seconds, and GPU memory/utilization when available.

## Remote Server Workflow

Server connection:

```bash
TERM=xterm-256color ssh -i ~/.ssh/vast -p 59644 root@74.48.140.178 -L 8080:localhost:8080
```

Remote setup should:

1. Pull the latest committed project code.
2. Restore or upload the current seed checkpoint, optimizer state, registry, and
   useful replay artifacts.
3. Verify CUDA availability with PyTorch and `nvidia-smi`.
4. Run tests or a quick import/compile check.
5. Start TensorBoard on port 8080.
6. Start the MVP run in a persistent session such as `tmux`.
7. Keep logs in `train.log`, `checkpoints/training_log.jsonl`, and
   `checkpoints/tensorboard`.

## Promotion Policy

Promotion remains stricter than smoke testing:

- At least 16 heuristic games.
- At least 16 previous-best games.
- No fixed-baseline Elo regression versus the champion.
- Zero runtime failures.

Short probes are diagnostics only. They must not select the remote seed.

## Decision Rules

After the first remote MVP run:

- If Elo and heuristic/previous-best scores improve, continue the same recipe
  for a longer second run.
- If training is slow because of MCTS, optimize search or batched inference
  before changing the model.
- If losses do not move, inspect replay/data quality before adding capacity.
- If value loss is noisy or draw-heavy, revise value targets or draw handling.
- If the model learns tactics but evaluation stays flat, improve evaluation and
  search alignment before scaling playouts.

The next stage is not to collect many more local experiments. It is to get one
watchable GPU MVP run, understand its bottleneck, and make the next change from
that evidence.
