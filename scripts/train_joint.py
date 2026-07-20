from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_uav import (  # noqa: E402
    AttentionChargingEnv, AttentionSACAgent, BalancedVariableReplayBuffer,
    EnergyModel, EnergyModelConfig, EpisodeRollout, JointEnvConfig, MAPPO, MAPPOConfig,
)
from joint_uav.low_env import VariableUAVEnv  # noqa: E402


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def make_energy(cfg: dict) -> EnergyModel:
    e = dict(cfg["energy"])
    e["airship_pos"] = tuple(e["airship_pos"])
    table_path = e.get("return_table_path")
    if table_path:
        p = Path(table_path)
        e["return_table_path"] = str(p if p.is_absolute() else ROOT / p)
    return EnergyModel(EnergyModelConfig(**e))


def make_train_system(env_cfg: dict, cfg: dict, device: str):
    low_env = VariableUAVEnv(env_cfg)
    initial = np.full(low_env.max_uavs, 0, dtype=np.int8)
    obs = low_env.reset(initial_status=initial, reset_dyn_rng=True)
    sac_cfg, model_cfg = cfg["sac"], cfg["model"]
    agent = AttentionSACAgent(
        global_dim=obs["global_obs"].shape[0],
        uav_obs_dim=obs["active_uav_obs"].shape[1],
        action_dim=int(model_cfg["action_dim"]),
        move_limit=float(env_cfg["uav"]["move_limit"]),
        model_cfg=model_cfg, sac_cfg=sac_cfg, device=device,
    )
    pretrained = cfg["training"].get("pretrained_sac_path")
    if pretrained:
        p = Path(pretrained)
        if not p.is_absolute():
            p = ROOT / p
        agent.load_pretrained(p)
        print(f"Loaded attention-SAC: {p}")
    replay = BalancedVariableReplayBuffer(
        int(sac_cfg["buffer_capacity"]), low_env.max_uavs, seed=int(cfg["seed"])
    )
    env = AttentionChargingEnv(
        low_env, agent, make_energy(cfg), JointEnvConfig.from_dict(cfg["joint"]), replay
    )
    upper_obs = env.reset(reset_dyn_rng=True)
    mappo = MAPPO(
        upper_obs["actor_obs"].shape[1], upper_obs["critic_obs"].shape[0],
        low_env.max_uavs, MAPPOConfig.from_dict(cfg["mappo"]), device,
    )
    return env, agent, replay, mappo


def pack_serving_trajectory_blocks(env: AttentionChargingEnv) -> tuple[np.ndarray, np.ndarray]:
    """Return [upper_block, UAV, low_point, xy] with NaN for non-serving UAVs."""
    blocks = env.history["trajectory_blocks"]
    points = env.cfg.block_steps + 1
    packed = np.full((len(blocks), env.n_uav, points, 2), np.nan, np.float32)
    serving = np.zeros((len(blocks), env.n_uav), bool)
    for b, tracks in enumerate(blocks):
        for i, traj in tracks.items():
            traj = np.asarray(traj, np.float32)
            if traj.shape != (points, 2):
                raise ValueError(
                    f"block {b} UAV{i} trajectory has {traj.shape}; expected {(points, 2)}"
                )
            packed[b, i] = traj
            serving[b, i] = True
    return packed, serving


def serving_path_lengths(env: AttentionChargingEnv) -> np.ndarray:
    packed, _ = pack_serving_trajectory_blocks(env)
    delta = np.diff(packed, axis=2)
    return np.nansum(np.linalg.norm(delta, axis=-1), axis=(0, 2)).astype(np.float32)


def collect_episode(
    env: AttentionChargingEnv, mappo: MAPPO, *, deterministic: bool,
    collect_replay: bool
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    obs = env.reset(reset_dyn_rng=False)
    rollout = EpisodeRollout()
    total_reward = 0.0
    done = False
    while not done:
        action, logp, value = mappo.act(
            obs["actor_obs"], obs["critic_obs"], deterministic=deterministic
        )
        next_obs, reward, done, _ = env.step(
            action, deterministic_low=deterministic, collect_low_replay=collect_replay
        )
        rollout.add(
            obs["actor_obs"], obs["critic_obs"], action, logp, reward, done, value,
            obs["agent_active"],
        )
        obs, total_reward = next_obs, total_reward + reward
    finished = rollout.finish(
        last_value=0.0, gamma=mappo.cfg.gamma, gae_lambda=mappo.cfg.gae_lambda
    )
    final_status = np.asarray(env.history["status"])[-1]
    path_length = float(serving_path_lengths(env).sum())
    return finished, {
        "return": float(total_reward),
        "buf": float(np.mean(env.history["buf_metric"])),
        "buf_max": float(np.max(env.history["buf_metric"])),
        "buf_min": float(np.min(env.history["buf_metric"])),
        "serving": float(np.mean(env.history["serving_count"])),
        "serving_sum": float(np.sum(env.history["serving_count"])),
        "dead": float(np.any(final_status == 3)),
        "dead_count": float(np.sum(final_status == 3)),
        "death_events": float(np.sum(env.history["newly_dead_count"])),
        "low_reward_mean": float(np.mean(env.low_env.hist_reward)),
        "low_team_max_sum_mean": float(np.mean(env.history["low_team_max_sum"])),
        "low_max_after_mean": float(np.mean(env.history["low_max_after"])),
        "oob_sum": float(np.sum(env.history["low_oob_count"])),
        "path_length_sum": path_length,
    }


def save_eval_episode(env: AttentionChargingEnv, path: Path, metric: dict[str, float]) -> None:
    """Save every useful upper/lower evaluation series; intentionally creates no figures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    upper_status = np.asarray(env.history["status"], dtype=np.int8)
    upper_battery = np.asarray(env.history["battery"], dtype=np.float32)
    low_status = np.asarray(env.low_env.status_history, dtype=np.int8)
    trajectory = np.stack(
        [np.asarray(x, dtype=np.float32) for x in env.low_env.traj], axis=0
    )
    trajectory_blocks, block_serving_mask = pack_serving_trajectory_blocks(env)
    path_lengths = serving_path_lengths(env)

    arrays = {
        "upper_buffer_metric": np.asarray(env.history["buf_metric"], np.float32),
        "upper_battery": upper_battery,
        "upper_status": upper_status,
        "upper_serving_count": np.asarray(env.history["serving_count"], np.int16),
        "upper_waiting_count": (upper_status[1:] == 1).sum(axis=1).astype(np.int16),
        "upper_charging_count": (upper_status[1:] == 2).sum(axis=1).astype(np.int16),
        "upper_dead_count": (upper_status[1:] == 3).sum(axis=1).astype(np.int16),
        "upper_newly_dead_count": np.asarray(env.history["newly_dead_count"], np.int16),
        "upper_queue_length": np.asarray(env.history["queue_length"], np.int16),
        "upper_reward": np.asarray(env.history["upper_reward"], np.float32),
        "upper_reward_serving": np.asarray(env.history["reward_serving"], np.float32),
        "upper_reward_waiting": np.asarray(env.history["reward_waiting"], np.float32),
        "upper_reward_buffer": np.asarray(env.history["reward_buffer"], np.float32),
        "upper_reward_death": np.asarray(env.history["reward_death"], np.float32),
        "low_block_reward": np.asarray(env.history["low_reward"], np.float32),
        "low_max_before": np.asarray(env.history["low_max_before"], np.float32),
        "low_max_after": np.asarray(env.history["low_max_after"], np.float32),
        "low_team_max_sum": np.asarray(env.history["low_team_max_sum"], np.float32),
        "low_covered_count": np.asarray(env.history["low_covered_count"], np.int16),
        "low_oob_count": np.asarray(env.history["low_oob_count"], np.int16),
        "low_serving_count": np.asarray(env.history["low_serving_count"], np.int16),
        "low_per_uav_collected_sum": np.asarray(
            env.history["low_per_uav_collected_sum"], np.float32
        ),
        "low_per_uav_collected_max": np.asarray(
            env.history["low_per_uav_collected_max"], np.float32
        ),
        "low_status": low_status,
        # The raw world history includes intentional airship/mission position
        # switches and must never be interpreted as a physical flight path.
        "raw_world_position_history": trajectory,
        "serving_trajectory_blocks": trajectory_blocks,
        "block_serving_mask": block_serving_mask,
        "per_uav_serving_path_length": path_lengths,
        "final_user_position": env.low_env.world.user_pos.astype(np.float32),
        "final_user_packets": env.low_env.world.user_pkts.astype(np.float32),
        "final_user_last_visit": env.low_env.world.user_last_visit.astype(np.float32),
        "final_uav_position": env.low_env.world.uav_pos.astype(np.float32),
        "mission_position_bank": env.mission_pos_bank.astype(np.float32),
    }
    arrays.update({f"summary_{k}": np.asarray(v, np.float32) for k, v in metric.items()})
    np.savez_compressed(path, **arrays)


def save_eval_plots(
    env: AttentionChargingEnv, eval_dir: Path, eval_episode: int,
    *, save_block_grids: bool = True,
) -> None:
    """Save evaluation figures without opening a GUI or calling plt.show()."""
    prefix = eval_dir / f"episode_{eval_episode:02d}"
    buf = np.asarray(env.history["buf_metric"], np.float32)
    battery = np.asarray(env.history["battery"], np.float32)[1:]
    status = np.asarray(env.history["status"], np.int8)[1:]
    serving = np.asarray(env.history["serving_count"], np.int16)
    steps = np.arange(len(buf))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(steps, buf, linewidth=1.8)
    ax.set(title="Mean System Max Buffer After Collection (10 Low Steps)",
           xlabel="Upper-level step", ylabel="Mean max buffer after")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{prefix}_buffer_metric.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for i in range(env.n_uav):
        ax.plot(np.arange(len(battery)), battery[:, i], label=f"UAV{i}", linewidth=1.4)
    ax.set(title="Battery Traces", xlabel="Upper-level step", ylabel="Battery level")
    ax.grid(True, alpha=0.3); ax.legend(loc="best", ncol=2)
    fig.tight_layout(); fig.savefig(f"{prefix}_battery_traces.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.step(steps, serving, where="post", linewidth=1.8)
    ax.set(title="UAVs Serving Count", xlabel="Upper-level step", ylabel="# SERVING")
    ax.set_ylim(-0.2, env.n_uav + 0.3); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{prefix}_serving_count.png", bbox_inches="tight"); plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=150)
    ax1.bar(steps, serving, alpha=0.30, label="Serving UAV count")
    ax1.set(xlabel="Upper-level step", ylabel="Serving UAV count")
    ax2 = ax1.twinx(); ax2.plot(steps, buf, color="tab:red", linewidth=1.5, label="Buffer metric")
    ax2.set_ylabel("Mean max buffer after")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left"); ax1.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{prefix}_serving_and_buffer.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    im = ax.imshow(status.T, aspect="auto", interpolation="nearest", vmin=0, vmax=3, cmap="viridis")
    ax.set(title="Upper-level UAV Status Schedule", xlabel="Upper-level step", ylabel="UAV ID")
    ax.set_yticks(np.arange(env.n_uav)); cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["SERVING", "WAITING", "CHARGING", "DEAD"])
    fig.tight_layout(); fig.savefig(f"{prefix}_status_schedule.png", bbox_inches="tight"); plt.close(fig)

    world = env.low_env.world
    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    ax.scatter(world.user_pos[:, 0], world.user_pos[:, 1], s=16, c="0.65", alpha=0.75, label="Sensors")
    ax.scatter(*world.airship_pos, marker="*", s=180, c="gold", edgecolors="black", label="Airship")
    colors = plt.get_cmap("tab10")
    labelled: set[int] = set()
    for tracks in env.history["trajectory_blocks"]:
        for i, tr in tracks.items():
            tr = np.asarray(tr, np.float32)
            ax.plot(
                tr[:, 0], tr[:, 1], linewidth=1.0, color=colors(i),
                label=f"UAV{i}" if i not in labelled else None,
            )
            labelled.add(i)
    ax.set(xlim=(0, world.world_size), ylim=(0, world.world_size), xlabel="x", ylabel="y",
           title="Serving-only UAV Trajectory Segments (No Teleport Connections)")
    ax.set_aspect("equal", adjustable="box"); ax.grid(True, alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{prefix}_uav_trajectories.png", bbox_inches="tight"); plt.close(fig)

    # Ten upper blocks per figure. A 100-block charging episode therefore
    # produces ten 2x5 grids and covers all 1000 low-level trajectory steps.
    block_groups = env.history["trajectory_blocks"]
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="0.35", markersize=5,
               label="Sensors"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="gold",
               markeredgecolor="black", markersize=12, label="Airship"),
    ] + [Line2D([0], [0], color=colors(i), lw=1.8, label=f"UAV{i}") for i in range(env.n_uav)]
    for group_start in range(0, len(block_groups), 10) if save_block_grids else ():
        group_end = min(group_start + 10, len(block_groups))
        fig, axes = plt.subplots(2, 5, figsize=(20, 8), dpi=140, squeeze=False)
        for local, ax in enumerate(axes.flat):
            b = group_start + local
            if b >= group_end:
                ax.axis("off")
                continue
            tracks = block_groups[b]
            ax.scatter(world.user_pos[:, 0], world.user_pos[:, 1], s=8, c="0.2", alpha=0.75)
            ax.scatter(*world.airship_pos, marker="*", s=90, c="gold", edgecolors="black", zorder=5)
            for i, tr in tracks.items():
                tr = np.asarray(tr, np.float32)
                color = colors(i)
                for xy in tr:
                    ax.add_patch(Circle(xy, env.low_env.coverage_radius, color=color,
                                        alpha=0.035, linewidth=0))
                ax.plot(tr[:, 0], tr[:, 1], color=color, linewidth=1.4)
                ax.scatter(tr[0, 0], tr[0, 1], s=22, marker="o", color=color,
                           edgecolors="black", linewidths=0.4, zorder=6)
                ax.scatter(tr[-1, 0], tr[-1, 1], s=25, marker="s", color=color,
                           edgecolors="black", linewidths=0.4, zorder=6)
            low_start, low_end = b * env.cfg.block_steps, (b + 1) * env.cfg.block_steps
            ids = ",".join(str(i) for i in sorted(tracks)) or "none"
            ax.set_title(f"Block {b} / low {low_start}-{low_end}\nserving: {ids}", fontsize=9)
            ax.set_xlim(0, world.world_size); ax.set_ylim(0, world.world_size)
            ax.set_aspect("equal", adjustable="box"); ax.grid(True, alpha=0.18)
        fig.legend(handles=legend_handles, loc="upper center", ncol=8, fontsize=9)
        fig.suptitle(
            f"Evaluation Serving Trajectories: Upper Blocks {group_start}-{group_end - 1}",
            y=0.985,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(
            f"{prefix}_trajectory_blocks_{group_start:03d}_{group_end - 1:03d}.png",
            bbox_inches="tight",
        )
        plt.close(fig)

    low_x = np.arange(len(env.history["low_max_before"]))
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(low_x, env.history["low_max_before"], label="max before", linewidth=1.2)
    ax.plot(low_x, env.history["low_max_after"], label="max after", linewidth=1.2)
    ax.set(title="Low-level Backlog", xlabel="Low-level step", ylabel="Packets")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(f"{prefix}_low_backlog.png", bbox_inches="tight"); plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=150)
    ax1.plot(low_x, env.history["low_team_max_sum"], label="team max sum", linewidth=1.2)
    ax1.set(xlabel="Low-level step", ylabel="Team max sum")
    ax2 = ax1.twinx(); ax2.plot(low_x, env.history["low_max_after"], color="tab:red",
                                alpha=0.8, label="system max buffer after")
    ax2.set_ylabel("System max buffer after")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right"); ax1.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{prefix}_low_collection.png", bbox_inches="tight"); plt.close(fig)


@torch.no_grad()
def evaluate(
    env_cfg: dict, cfg: dict, agent: AttentionSACAgent, mappo: MAPPO, episodes: int,
    *, training_upper_step: int, out_dir: Path,
) -> dict[str, float]:
    metrics = []
    eval_dir = out_dir / "evaluations" / f"step_{training_upper_step:07d}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(episodes):
        ecfg = copy.deepcopy(env_cfg)
        ecfg["seeds"]["world_dyn_seed"] += 100_000 + ep
        low_env = VariableUAVEnv(ecfg)
        env = AttentionChargingEnv(
            low_env, agent, make_energy(cfg), JointEnvConfig.from_dict(cfg["joint"]), replay=None
        )
        _, m = collect_episode(env, mappo, deterministic=True, collect_replay=False)
        metrics.append(m)
        save_eval_episode(env, eval_dir / f"episode_{ep:02d}.npz", m)
        save_eval_plots(env, eval_dir, ep, save_block_grids=(ep == 0))
        print(
            f"  [eval episode {ep}] return={m['return']:.3f} avg_buf={m['buf']:.3f} "
            f"sum serving count over time: {int(m['serving_sum'])} "
            f"dead={int(m['dead_count'])} oob={int(m['oob_sum'])} "
            f"path={m['path_length_sum']:.1f}"
        )

    summary: dict[str, float] = {"training_upper_step": float(training_upper_step)}
    for key in metrics[0]:
        values = np.asarray([x[key] for x in metrics], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())
    (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        eval_dir / "episode_summaries.npz",
        **{key: np.asarray([x[key] for x in metrics], np.float32) for key in metrics[0]},
    )
    return summary


def save_checkpoint(
    path: Path, training_episode: int, training_upper_step: int,
    agent: AttentionSACAgent, mappo: MAPPO, cfg: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "training_episode": training_episode, "training_upper_step": training_upper_step,
        "attention_sac": agent.state_dict(),
        "mappo": mappo.state_dict(), "config": cfg,
    }, path)


class TrainingRolloutCollector:
    """Collect an exact number of upper steps while carrying partial episodes across PPO updates."""

    def __init__(self, env: AttentionChargingEnv, mappo: MAPPO) -> None:
        self.env, self.mappo = env, mappo
        self.obs: dict[str, np.ndarray] | None = None
        self.rollout: EpisodeRollout | None = None
        self.completed_episodes = 0
        self.total_upper_steps = 0
        self.ep_return = 0.0

    def _start_episode(self) -> None:
        self.obs = self.env.reset(reset_dyn_rng=False)
        self.rollout = EpisodeRollout()
        self.ep_return = 0.0

    def collect(
        self, target_steps: int, max_completed_episodes: int
    ) -> tuple[list[dict[str, np.ndarray]], list[dict[str, float]], int]:
        chunks: list[dict[str, np.ndarray]] = []
        completed_metrics: list[dict[str, float]] = []
        steps = 0

        while steps < target_steps and self.completed_episodes < max_completed_episodes:
            if self.obs is None:
                self._start_episode()
            assert self.obs is not None and self.rollout is not None

            action, logp, value = self.mappo.act(
                self.obs["actor_obs"], self.obs["critic_obs"], deterministic=False
            )
            next_obs, reward, done, _ = self.env.step(
                action, deterministic_low=False, collect_low_replay=True
            )
            self.rollout.add(
                self.obs["actor_obs"], self.obs["critic_obs"], action, logp,
                reward, done, value, self.obs["agent_active"],
            )
            self.obs, self.ep_return = next_obs, self.ep_return + reward
            steps += 1
            self.total_upper_steps += 1

            if done:
                chunks.append(self.rollout.finish(
                    last_value=0.0, gamma=self.mappo.cfg.gamma,
                    gae_lambda=self.mappo.cfg.gae_lambda,
                ))
                completed_metrics.append({
                    "return": float(self.ep_return),
                    "buf": float(np.mean(self.env.history["buf_metric"])),
                    "serving": float(np.mean(self.env.history["serving_count"])),
                    "serving_sum": float(np.sum(self.env.history["serving_count"])),
                    "dead": float(np.any(np.asarray(self.env.history["status"])[-1] == 3)),
                })
                self.completed_episodes += 1
                self.obs, self.rollout = None, None

        # PPO rollout boundary, not an environment terminal: bootstrap and carry state onward.
        if self.obs is not None and self.rollout is not None and self.rollout.rewards:
            chunks.append(self.rollout.finish(
                last_value=self.mappo.value(self.obs["critic_obs"]),
                gamma=self.mappo.cfg.gamma, gae_lambda=self.mappo.cfg.gae_lambda,
            ))
            self.rollout = EpisodeRollout()
        return chunks, completed_metrics, steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "joint.yaml")
    parser.add_argument("--env-config", type=Path, default=ROOT / "configs" / "env.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny end-to-end check")
    args = parser.parse_args()

    cfg, env_cfg = load_cfg(args.config), load_cfg(args.env_config)
    if args.smoke:
        cfg["training"].update(
            num_episodes=1, rollout_steps=2, mappo_warmup_upper_steps=2,
            eval_interval_upper_steps=1, eval_start_upper_steps=1, eval_episodes=1,
            checkpoint_interval_upper_steps=1, sac_update_start_size=10_000,
            sac_updates_per_rollout=1,
        )
        cfg["joint"]["episode_len_steps"] = 2
        env_cfg["time"]["high_steps"] = 2
    seed_all(int(cfg["seed"]))
    env, agent, replay, mappo = make_train_system(env_cfg, cfg, args.device)
    tr = cfg["training"]
    out_dir = ROOT / tr["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    logs: list[dict[str, float]] = []
    best_return = -np.inf
    collector = TrainingRolloutCollector(env, mappo)
    total_episodes = int(tr["num_episodes"])
    target_upper_steps = total_episodes * int(cfg["joint"]["episode_len_steps"])
    rollout_steps = int(tr["rollout_steps"])
    update_idx = 0
    next_eval = max(
        int(tr["eval_interval_upper_steps"]), int(tr.get("eval_start_upper_steps", 0))
    )
    next_checkpoint = int(tr["checkpoint_interval_upper_steps"])
    train_start_time = time.perf_counter()

    while collector.completed_episodes < total_episodes:
        update_start_time = time.perf_counter()
        update_idx += 1
        rollout_start_upper_step = collector.total_upper_steps
        # MAPPO and SAC are both frozen for this entire on-policy rollout.
        chunks, train_metrics, collected_steps = collector.collect(
            rollout_steps, total_episodes
        )
        progress = collector.total_upper_steps / max(1, target_upper_steps)
        # Warm-up is step based, so episode length and death cannot change it.
        if rollout_start_upper_step >= int(tr["mappo_warmup_upper_steps"]):
            ppo_info = mappo.update(chunks, progress=progress)
        else:
            ppo_info = {
                "actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0,
                "kl": 0.0, "clipfrac": 0.0, "entropy_coef": 0.0,
                "samples": float(collected_steps), "early_stop": 0.0,
            }

        sac_losses = []
        if len(replay) >= int(tr["sac_update_start_size"]):
            for _ in range(int(tr["sac_updates_per_rollout"])):
                batch = replay.sample(
                    int(cfg["sac"]["batch_size"]), args.device,
                    float(tr["replay_balanced_fraction"]),
                )
                sac_losses.append(agent.update(batch))

        # At most one evaluation per changed policy; intervals are upper-step based.
        eval_payload: dict[str, float] = {}
        if collector.total_upper_steps >= next_eval:
            eval_label = collector.total_upper_steps
            ev = evaluate(
                env_cfg, cfg, agent, mappo, int(tr["eval_episodes"]),
                training_upper_step=eval_label, out_dir=out_dir,
            )
            eval_payload.update({f"eval_{k}": v for k, v in ev.items()})
            if ev["return_mean"] > best_return and ev["dead_mean"] == 0.0:
                best_return = ev["return_mean"]
                save_checkpoint(
                    out_dir / "best.pt", collector.completed_episodes,
                    collector.total_upper_steps, agent, mappo, cfg,
                )
            interval = int(tr["eval_interval_upper_steps"])
            next_eval = (collector.total_upper_steps // interval + 1) * interval

        row = {
            "update": float(update_idx),
            "completed_episodes": float(collector.completed_episodes),
            "total_upper_steps": float(collector.total_upper_steps),
            "rollout_upper_steps": float(collected_steps),
            "train_return": float(np.mean([m["return"] for m in train_metrics])) if train_metrics else 0.0,
            "train_buf": float(np.mean([m["buf"] for m in train_metrics])) if train_metrics else 0.0,
            "train_serving": float(np.mean([m["serving"] for m in train_metrics])) if train_metrics else 0.0,
            "replay_size": float(len(replay)), **{f"ppo_{k}": v for k, v in ppo_info.items()},
            **eval_payload,
        }
        if sac_losses:
            row.update(
                sac_critic_loss=float(np.mean([x.critic_loss for x in sac_losses])),
                sac_actor_loss=float(np.mean([x.actor_loss for x in sac_losses])),
                alpha=float(sac_losses[-1].alpha),
            )

        logs.append(row)
        now = time.perf_counter()
        update_seconds = now - update_start_time
        elapsed_seconds = now - train_start_time
        remaining_steps = max(0, target_upper_steps - collector.total_upper_steps)
        eta_seconds = (
            elapsed_seconds / collector.total_upper_steps * remaining_steps
            if collector.total_upper_steps > 0 else 0.0
        )
        print(
            f"[update {update_idx:04d} | ep {collector.completed_episodes:04d}/{total_episodes} "
            f"| upper_step {collector.total_upper_steps:07d}] "
            f"R={row['train_return']:.3f} buf={row['train_buf']:.2f} "
            f"serve={row['train_serving']:.2f} replay={len(replay)} "
            f"kl={row['ppo_kl']:.4f} alpha={agent.alpha.detach().item():.3f} "
            f"time={format_duration(update_seconds)} "
            f"elapsed={format_duration(elapsed_seconds)} eta={format_duration(eta_seconds)}"
        )
        (out_dir / "metrics.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
        if collector.total_upper_steps >= next_checkpoint:
            save_checkpoint(
                out_dir / f"step_{collector.total_upper_steps:07d}.pt",
                collector.completed_episodes, collector.total_upper_steps,
                agent, mappo, cfg,
            )
            while next_checkpoint <= collector.total_upper_steps:
                next_checkpoint += int(tr["checkpoint_interval_upper_steps"])

    save_checkpoint(
        out_dir / "final.pt", collector.completed_episodes,
        collector.total_upper_steps, agent, mappo, cfg,
    )
    print(f"Training finished. Total time: {format_duration(time.perf_counter() - train_start_time)}")


if __name__ == "__main__":
    main()
