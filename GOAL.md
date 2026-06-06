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

Completed MVP result on 2026-06-06:

- Final checkpoint: `large_16x16_top_human_gpu_final_g524_20260606_014356_758050`.
- Self-play: 512/512 games, 35,388 moves, 0 draws, 0 illegal/runtime eval failures.
- Throughput: about 3.24 self-play moves/sec and 2.80 search moves/sec over
  the parallel stream.
- Efficiency diagnosis: self-play was MCTS/search-coordination-bound. GPU
  inference batching worked, but utilization came in bursts; CPU use stayed far
  below all-core saturation. External evaluation was serial CPU-bound and much
  slower than expected.
- Built-in final eval: Elo 454, 32/32 vs random, 3.125% vs heuristic.
- Promotion eval: Elo 684, 21.875% vs heuristic, 9.375% vs previous best; no
  promotion.
- Replay inspection is now reproducible with `replay_audit.py`. The 50k-sample
  MVP replay tail classified as `diffuse_policy_targets`: value labels were
  balanced and draw-free, 84.0% of policy targets had max probability <= 0.02,
  only 13.8% had max probability >= 0.25, and normalized policy entropy
  averaged 0.795.
- Replay target rewrites can now be probed without training by adding
  `--probe-transforms` to `replay_audit.py`; the probe compares power, top-k,
  min-prob, top-k+power, and top-1 targets by retained MCTS mass, support size,
  top-1 agreement, max-prob/entropy deltas, and distortion metrics.
- The top-human GPU preset now applies the focused self-play target rewrite
  selected from that probe: `self_play_target_transform=top_k` with
  `self_play_target_top_k=16`. The 50k replay probe showed this raises mean
  policy max probability from about 0.147 to 0.207, drops normalized entropy
  by about 0.366, preserves the search top move, and retains about 24.8% of
  original MCTS target mass before renormalization.
- Decision: do not continue this recipe longer. Improve evaluation/search
  efficiency and target sharpness/alignment before another long run.
- Integrated audit decision:
  `decision_recommendation.label=fix_search_target_alignment_before_scaling`.

Before the next remote training run:

- Keep the draw-penalty champion as the seed unless a stricter promotion probe
  replaces it.
- Use the parallel external evaluator added after the MVP (`eval_parallel_games`
  in the preset and repaired checkpoint metadata) so previous-best checks do
  not run as one-core CPU jobs.
- Do not rerun `large_16x16_top_human_gpu` unchanged. The next recipe should
  change search/target alignment, such as filtering or sharpening MCTS policy
  targets, improving root target quality, or reducing duplicate serial eval.
- Preserve TensorBoard/JSONL instrumentation, especially eval scores, replay
  samples, target entropy/sharpness if added, and runtime timings.
- Verify required TensorBoard scalar coverage with `tensorboard_audit.py`
  after each long run.
- Before a long run, check the monitor/CUDA/parallel-worker stack with
  `remote_health_check.py --preset large_16x16_top_human_gpu --eval-games 32`.
- After each long run, summarize the exact JSONL run slice with
  `training_audit.py --gpu-log gpu_smi.log --resource-log resource_monitor.jsonl --replay-path checkpoints/replay_16x16_n5_r4_f64.pkl`
  instead of mixing aborted attempts with the completed checkpoint run.
  The audit's `bottleneck_assessment` is the source of truth for deciding
  whether the next change should target GPU batching, CPU utilization, or MCTS
  search coordination, and `replay_quality` is the source of truth for target
  sharpness/value-label quality.
- Before changing the search target recipe, summarize replay quality with
  `replay_audit.py checkpoints/replay_16x16_n5_r4_f64.pkl --max-samples 50000 --strategy tail`.
  If the audit again reports diffuse policy targets, fix target sharpness or
  search alignment before scaling.
- Before changing the target rewrite again, run the same audit with
  `--probe-transforms` and choose from transform evidence rather than adding
  another heuristic by guesswork.
- The current chosen rewrite is top-k-16 self-play target sharpening. Do not
  add more target heuristics before evaluating whether this improves target
  entropy and fixed-baseline scores with the existing efficiency instrumentation.
- New runs also emit self-play target-quality scalars directly, including
  `self_play/policy_target_diffuse_fraction`,
  `self_play/policy_target_normalized_entropy_mean`, and
  `self_play/value_target_draw_fraction`.
- For future remote runs, start `resource_monitor.py` in tmux so CPU load,
  cgroup CPU-quota utilization, memory, and GPU utilization are sampled as
  JSONL. Treat `cpu_util_percent` as the training container's CPU usage and
  `host_cpu_util_percent` as ambient host-wide pressure. The monitor also
  mirrors those resource scalars into
  `checkpoints/tensorboard/resource_monitor` by default.
  The current resource monitor started after the 2026-06-06 MVP run, so that
  completed run still uses `gpu_smi.log` plus training JSONL as its resource
  evidence; future runs should compare in-run CPU samples against the idle
  baseline from `resource_monitor.jsonl`.
- The `large_16x16_top_human_gpu` preset now uses auto self-play/eval
  parallelism capped to the usable cgroup CPU worker count. On the current
  remote GPU box, the visible CPU count is 72 but the cgroup quota is 11.52
  cores, so auto resolves to 12 workers unless explicitly overridden.

Final server wrap on 2026-06-06 before the rented GPU expired:

- Final health check passed: PyTorch CUDA saw one NVIDIA GeForce RTX 4060,
  `nvidia-smi` was available, TensorBoard HTTP returned 200, and the required
  training/resource TensorBoard scalar tags were present.
- No training process was still active at wrap time; GPU memory was 0 MiB used
  and utilization was 0%. The active tmux services were monitors plus
  TensorBoard/web.
- The final compact-request smoke checkpoint was
  `large_16x16_top_human_gpu_final_g24_20260606_171202_292769`, not promoted
  and not a new seed. Its purpose was to validate instrumentation and IPC
  changes after the MVP run.
- The compact-request smoke produced 12 self-play games, 915 moves, and about
  6.37 self-play moves/sec. JSONL/TensorBoard recorded zero draw self-play,
  zero invalid replay samples, and small loss decreases over the smoke
  (`loss` -0.281, `policy_loss` -0.234, `value_loss` -0.047).
- Runtime evidence from that smoke still points to search coordination/MCTS as
  the bottleneck: stream elapsed was 143.7s, worker search-time sum was
  1245.3s, worker wait was 57.9s, training was 8.3s, eval was 20.6s,
  checkpointing was 8.9s, replay save was 5.5s, and total JSONL/TensorBoard
  logging was below 1s.
- Compact response and compact request protocols are now enabled by default in
  the top-human GPU preset. Request state tensors are sent as `uint8`, reducing
  request state payloads to 1024 bytes/position on 16x16 boards; compact
  responses send flat probability/value arrays.
- Compact IPC reduced request/response overhead but did not fully solve
  utilization. The next efficiency work should target MCTS request cadence,
  coalescing, and batched inference/search scheduling before adding more
  tactical heuristics.
- The remote artifacts under `checkpoints/` were about 1.9 GB and remain
  ignored by git. To continue exact training state elsewhere, back up or
  restore the seed checkpoint, optimizer, registry, replay pickle, JSONL, and
  TensorBoard directory outside the source-code push.

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
.venv/bin/tensorboard --logdir checkpoints/tensorboard --host 0.0.0.0 --port 8081
```

The MVP remote TensorBoard session currently listens on port 8081. The local
tunnel for viewing it is:

```bash
TERM=xterm-256color ssh -i ~/.ssh/vast -p 59644 root@74.48.140.178 -L 8080:localhost:8081
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
TERM=xterm-256color ssh -i ~/.ssh/vast -p 59644 root@74.48.140.178 -L 8080:localhost:8081
```

Remote setup should:

1. Pull the latest committed project code.
2. Restore or upload the current seed checkpoint, optimizer state, registry, and
   useful replay artifacts.
3. Verify CUDA availability with PyTorch and `nvidia-smi`.
4. Run tests or a quick import/compile check.
5. Start TensorBoard on remote port 8081 and forward it to local port 8080.
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
