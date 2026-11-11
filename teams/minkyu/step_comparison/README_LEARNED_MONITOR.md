# Learned generator: 500-episode distribution history

This is a **logging-only addition** to the existing online/frozen Learned
implementation. It does not modify the learned architecture, supervised labels,
rollout policy, TD targets, main-model evaluation, or `uniform/rule_based/observed`.
No C++ rebuild is needed. Copy the supplied files into the existing Ama repo,
preserving all other repo files.

## Existing training runs do not need to be discarded

With the previously used `--save-every 500`, `puyo_train.py` already saves
`learned_ep_000500.pt`, `learned_ep_001000.pt`, etc. **If they exist, those
checkpoints contain the entire learned model at those epochs**. Generate the
historical distributions and heldout diagnostics in a new directory, without
restarting RL training:

```bash
python -m teams.minkyu.step_comparison.gtr.learned_monitor_backfill \
  --run-dir teams/minkyu/step_comparison/runs/n15_learned_online_20k \
  --reference-games 512 --reference-seed-start 700000
```

The backfill also follows accessible `parent_resume_checkpoint` chains if a
previous run was interrupted and resumed into a new run directory. It refuses
to overwrite an existing nonempty analysis folder. Existing model weights,
optimizers, run logs, and evaluation files are NOT modified. It can only
reconstruct epochs for which `learned_ep_*.pt` exists. For a fresh
non-pretrained model, the epoch-0 distribution is mathematically known to be
Uniform because `PairNet` starts with zero final logits; the initial **hidden
weights** are not reconstructed.

If a training process is still running, you do not need to stop it just to
collect the distribution history; run backfill after it finishes. If you have
already pressed Ctrl+C, your most recent *completed* `last_resume.pt` can be
used to continue in a **different** run directory using the previous full
training command plus `--resume <old_run>/last_resume.pt`.

## New training runs: automatic snapshots

With the patch applied, online/frozen learned runs default to:

```
--learned-monitor-every 500
--learned-monitor-games 512
--learned-monitor-seed-start 700000
```

Training also writes a baseline snapshot at episode 0 and the final episode
if it is not divisible by 500. The fixed reference uses actual newly public
NEXT2 observations from 512 independent Ama engine seeds, **never true hidden
pairs while making a rollout**. It does NOT feed reference examples to the
supervised optimizer. Those reference seeds must remain separate from the
actor's training/validation seeds and any offline generator training datasets.

### Output files

- `learned_distribution_snapshots.csv`: for each episode and each fixed public
  `turn, CURRENT, NEXT, NEXT2` probe, all **16** categorical probabilities.
- `learned_position_distributions.csv`: for each episode and each position with
  enough heldout observations, 16 predicted average probabilities versus 16
  empirical observed Ama probabilities, and position-wise KL/TV.
- `learned_distribution_metrics.csv`: heldout NLL, categorical top-1 accuracy,
  NLL improvement over Uniform (`log(16)`), weighted position-marginal KL/TV,
  observations, optimizer updates, and batched CPU inference microseconds per
  reference context. Batched timing is NOT single-pair rollout latency.
- `learned_monitor_reference.npz` / `.json`: the fixed reference examples and
  public probe contexts, including seed range and evaluation caveats.
- `learned_ep_000000.pt`, `learned_ep_000500.pt`, etc.: existing or newly
  saved learned weights at each monitored epoch.

All probabilities and metrics are stored as numbers (not charts), allowing
later plots without rerunning training. Each comparison uses the same fixed
probes/reference across the checkpoints of a run.

**Statistics caveat:** KL and TV above compare the position-marginal empirical
observed distribution against predicted probabilities *averaged over the fixed
reference histories at that position*. They are **not** KL/TV to the exact
conditional Ama distribution given CURRENT/NEXT/NEXT2. The reference uses
surviving game observations from a deterministic engine-play policy, which can
induce selection bias at later positions. Heldout NLL is the more useful
conditional predictive-quality metric, and the full 16-category per-context
probability history is logged separately. These diagnostics must not be used
as final independent test data for selecting RL hyperparameters.

### Tests

```bash
python -m pytest -q teams/minkyu/step_comparison/tests
python -m teams.minkyu.step_comparison.puyo_train --help
```

Use a new `--run-dir` for a fresh run. The trainer refuses to start if its
run directory is already nonempty, to prevent accidental overwrites.
