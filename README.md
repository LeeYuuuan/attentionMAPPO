# Attention-SAC + MAPPO joint UAV training

This project replaces the fixed-size low-level SAC zoo with one variable-length
attention-SAC while preserving the original charging action meanings, charging
slot rules, and FIFO waiting queue.

## Run

```bash
pip install -r requirements.txt
python scripts/train_joint.py --smoke
python scripts/train_joint.py
```

`training.pretrained_sac_path` is optional. Leave it as `null` for one complete
from-scratch joint run. MAPPO and attention-SAC are both trained from their
first eligible update; no old checkpoint is required.

## Training order

The MAPPO behavior policy stays frozen for `rollout_steps: 100` upper-level steps.
The lower attention-SAC uses its original 5,000-transition safe-random warm-up.
After that, every eligible low-level transition is stored and immediately
followed by one replay update before the next low action is selected.
The rollout may cross episode boundaries and carries partial episodes across
updates. True terminals cut GAE; a PPO rollout boundary bootstraps from the
centralized critic. After collection, MAPPO updates once. The default run ends after 4000 completed
training episodes. One episode contains 100 upper decisions and every upper
decision fixes the serving set for 10 low-level trajectory steps. The default
`end_if_any_dead: true` terminates and resets immediately after the first death,
so a failed episode is shorter than 100 upper decisions. Training still ends
at exactly 4000 completed episodes.

After the 5,000-transition warm-up, SAC performs one update per newly stored low
transition with batch size 64, fixed alpha 0.2, and
actor/critic learning rate 1e-4. Evaluation never writes replay or updates SAC.
MAPPO updates after each
100-step on-policy rollout. This matches the old loop's effective rollout size:
although its configured cap was 2048, the environment terminated at no more
than 100 steps and PPO updated after that episode. If death ends an episode
early, its terminal trajectory is retained and collection continues in a new
episode until the 100-transition rollout is full. Its tuned upper-level settings match the old run:
four 512-wide ReLU actor layers, three 512-wide ReLU critic layers, clip 0.2,
10 epochs, 2048-sample minibatches, entropy coefficient 0.001, actor/critic
gradient limits 0.5/2.0, and value coefficient 0.5. Already-dead UAV actions
are masked out of the MAPPO actor loss.

The low reward is the sum of the maximum buffer received by each serving UAV,
minus the system maximum buffer after collection and OOB penalties. The upper
reward uses weighted normalized serving and waiting counts, minus the mean
system maximum buffer after collection over the ten-step block, with a one-time
newly-dead penalty. Current weights are serving 3.0, waiting 1.0, buffer 1.0,
and newly-dead 10.0.

## 50-episode reports and evaluation files

Every 50 completed training episodes, the script prints one upper/lower metric
summary and runs exactly one deterministic evaluation episode. It also updates
`training_low_max_buffer_so_far.png` and its NPZ data file. Evaluation folders
are labelled by completed training episode, for example:

```text
outputs/joint_attention_mappo_v4/evaluations/episode_0050/
  summary.json
  episode_summaries.npz
  episode_00.npz
  episode_01.npz
  episode_02.npz
  episode_00_buffer_metric.png
  episode_00_battery_traces.png
  episode_00_serving_count.png
  episode_00_serving_and_buffer.png
  episode_00_status_schedule.png
  episode_00_uav_trajectories.png
  episode_00_trajectory_blocks_000_009.png
  ...
  episode_00_trajectory_blocks_090_099.png
  episode_00_low_backlog.png
  episode_00_low_collection.png
```

Matplotlib uses the non-interactive `Agg` backend. Figures are written with
`savefig()` and closed immediately; training never calls `plt.show()`.

Each episode file contains upper-level buffer, battery, status, serving/waiting/
charging counts, queue length and rewards. It also stores all low-level backlog,
coverage, collection, OOB and serving series, complete UAV trajectories, path
lengths, final sensor buffers and final UAV/mission positions. Console output
includes `sum serving count over time` for every evaluation episode.
The NPZ stores serving trajectories as `[100, N_UAV, 11, 2]` blocks with NaN
for non-serving UAVs. Trajectory figures never connect an airship/mission state
switch. Detailed 2x5 block grids are saved only for eval episode 0 to keep
periodic evaluation from producing thousands of large images; episodes 1/2
retain the complete block arrays in NPZ.

## Preserved charging semantics

| status | action 0 | action 1 |
|---|---|---|
| SERVING | request charging | keep serving |
| WAITING | keep waiting | leave and serve |
| CHARGING | keep charging | leave and serve |

The implementation keeps the old transition order: release charging UAVs,
allow waiting UAVs to leave, fill slots from FIFO, then process new requests in
ascending UAV-ID order. A full UAV is automatically released to `SERVING` and
cannot use its old `CHARGING` action again as a new same-step charge request.

## Important implementation choices

- The packet world resets once per episode, never once per upper-level block.
- Packet dynamics continue even with zero serving UAVs.
- A separate mission-position bank preserves return/outbound energy locations.
- The supplied `return_energy_table.npz` is loaded from
  `checkpoints/energy_tables/`; its radius-zero entry retains vertical return
  energy between UAV and airship altitudes.
- Each low step is 10 seconds of movement plus 50 seconds of collection hover.
  Block energy charges horizontal/hover power for the movement phase and hover
  power for the full collection phase; ten low steps equal one 10-minute upper
  charging decision.
- `python tools/gen_return_energy_table.py` regenerates the same table format
  after energy parameters are changed.
- A low-level transition is terminal at every upper-block boundary, avoiding an
  invalid SAC bootstrap across an unseen upper-level schedule change.
- MAPPO keeps the old tuned network and PPO hyperparameters, while fixing the
  old off-by-one terminal mask in GAE. No value normalization, value clipping,
  KL early stop, entropy schedule, or learning-rate schedule is enabled by
  default because those were not part of the old effective configuration.
- SAC uses mask-dependent target entropy and active-count-balanced replay.
