from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import yaml

from joint_uav import (
    AttentionChargingEnv, AttentionSACAgent, EnergyModel, EnergyModelConfig,
    EpisodeRollout, JointEnvConfig,
)
from joint_uav.low_env import VariableUAVEnv


ROOT = Path(__file__).resolve().parents[1]


def make_env(*, n_uav: int = 4, episode_len: int = 4, coverage: float = 86.6):
    env_cfg = yaml.safe_load((ROOT / "configs" / "env.yaml").read_text())
    env_cfg = copy.deepcopy(env_cfg)
    env_cfg["uav"]["max_uavs"] = n_uav
    env_cfg["uav"]["coverage_radius"] = coverage
    env_cfg["time"]["high_steps"] = episode_len
    env_cfg["time"]["low_steps_per_high"] = 1
    low = VariableUAVEnv(env_cfg)
    obs = low.reset(initial_status=np.zeros(n_uav, np.int8), reset_dyn_rng=True)
    model_cfg = {"hidden_dim": 16, "n_heads": 4, "n_layers": 1}
    sac_cfg = {"actor_lr": 1e-4, "critic_lr": 1e-4, "alpha_lr": 1e-4}
    agent = AttentionSACAgent(
        global_dim=obs["global_obs"].shape[0], uav_obs_dim=obs["active_uav_obs"].shape[1],
        action_dim=2, move_limit=env_cfg["uav"]["move_limit"], model_cfg=model_cfg,
        sac_cfg=sac_cfg, device="cpu",
    )
    joint = AttentionChargingEnv(
        low, agent, EnergyModel(EnergyModelConfig()),
        JointEnvConfig(episode_len_steps=episode_len, block_steps=1, max_slots=2, max_wait=2),
    )
    joint.reset(reset_dyn_rng=True)
    return joint


def test_original_fifo_order_and_context_actions():
    env = make_env(n_uav=4)
    # All four request charging: IDs 0/1 take slots, IDs 2/3 enter FIFO.
    env.step(np.array([0, 0, 0, 0]), deterministic_low=True, collect_low_replay=False)
    assert env._status().tolist() == [2, 2, 1, 1]
    assert env.wait_queue == [2, 3]

    # Full charging UAVs are released first; FIFO then fills slots with 2/3.
    env.step(np.array([0, 0, 0, 0]), deterministic_low=True, collect_low_replay=False)
    assert env._status().tolist() == [0, 0, 2, 2]
    assert env.wait_queue == []


def test_full_charging_uav_is_released_and_old_zero_is_not_reused():
    env = make_env(n_uav=2)
    # Both enter CHARGING and become full_pending at the end of this block.
    env.step(np.array([0, 0]), deterministic_low=True, collect_low_replay=False)
    assert env._status().tolist() == [2, 2]
    assert all(u.full_pending for u in env.uavs)

    # These zeros were selected in the CHARGING context (keep charging).  Full
    # UAVs are auto-released; the same zeros cannot become new charge requests.
    env.step(np.array([0, 0]), deterministic_low=True, collect_low_replay=False)
    assert env._status().tolist() == [0, 0]
    assert not any(u.full_pending for u in env.uavs)


def test_waiting_action_one_leaves_queue():
    env = make_env(n_uav=4)
    env.step(np.array([0, 0, 0, 0]), deterministic_low=True, collect_low_replay=False)
    env.uavs[0].full_pending = False
    env.uavs[1].full_pending = False
    # UAV2 uses 1 while WAITING, so it leaves; UAV3 stays first in FIFO.
    env.step(np.array([0, 0, 1, 0]), deterministic_low=True, collect_low_replay=False)
    assert env.uavs[2].status == 0
    assert env.wait_queue == [3]


def test_world_advances_with_zero_serving_and_does_not_reset_per_block():
    env = make_env(n_uav=4, coverage=1e-6)
    start_now = env.low_env.world.now
    start_visit = env.low_env.world.user_last_visit.copy()
    env.step(np.array([0, 0, 0, 0]), deterministic_low=True, collect_low_replay=False)
    # Two charging + two waiting => zero serving, but one 60-second low step still occurs.
    assert env.low_env.world.now == start_now + 60.0
    np.testing.assert_allclose(env.low_env.world.user_last_visit, start_visit + 60.0)


def test_gae_uses_current_transition_done_mask():
    r = EpisodeRollout()
    obs_a = np.zeros((2, 3), np.float32)
    obs_c = np.zeros(4, np.float32)
    for reward, done in [(1.0, False), (1.0, True)]:
        r.add(obs_a, obs_c, np.zeros(2), np.zeros(2), reward, done, 0.0)
    out = r.finish(last_value=123.0, gamma=1.0, gae_lambda=1.0)
    np.testing.assert_allclose(out["advantages"], np.array([2.0, 1.0], np.float32))


def test_attention_agent_accepts_one_and_many_uavs():
    env = make_env(n_uav=4)
    for ids in ([2], [0, 1, 2, 3]):
        status = np.full(4, 1, np.int8); status[list(ids)] = 0
        obs = env.low_env.set_uav_status(status)
        action, returned_ids = env.low_agent.select_action(obs, deterministic=False)
        assert action.shape == (len(ids), 2)
        assert returned_ids.tolist() == list(ids)
        assert torch.isfinite(action).all()


def test_fixed_horizon_does_not_end_when_a_uav_dies():
    env = make_env(n_uav=4, episode_len=4)
    env.cfg.end_if_any_dead = False
    env.uavs[0].battery = 1e-8
    _, _, done, info = env.step(
        np.ones(4, np.int64), deterministic_low=True, collect_low_replay=False
    )
    assert not done
    assert info["newly_dead_count"] == 1
    _, _, done, info = env.step(
        np.ones(4, np.int64), deterministic_low=True, collect_low_replay=False
    )
    assert not done
    assert info["newly_dead_count"] == 0


def test_end_if_any_dead_terminates_immediately():
    env = make_env(n_uav=4, episode_len=4)
    env.cfg.end_if_any_dead = True
    env.uavs[0].battery = 1e-8
    _, _, done, info = env.step(
        np.ones(4, np.int64), deterministic_low=True, collect_low_replay=False
    )
    assert done
    assert info["newly_dead_count"] == 1
    assert len(env.history["battery"]) == 2  # reset plus the fatal upper step


def test_return_table_keeps_vertical_energy_at_zero_radius():
    from joint_uav import EnergyModel, EnergyModelConfig

    table = ROOT / "checkpoints" / "energy_tables" / "return_energy_table.npz"
    model = EnergyModel(EnergyModelConfig(return_table_path=str(table)))
    energy = model.return_energy_frac(np.array([200.0, 200.0], np.float32))
    assert energy > 0.0
    np.testing.assert_allclose(energy, model.frac[0], rtol=1e-6)


def test_block_energy_uses_ten_second_move_and_fifty_second_hover():
    model = EnergyModel(EnergyModelConfig())
    moving = np.stack([np.arange(11), np.zeros(11)], axis=1).astype(np.float32)
    hovering = np.zeros((11, 2), np.float32)
    e_move = model.block_energy_frac(moving, move_sec=10.0, collect_sec=50.0)
    e_hover = model.block_energy_frac(hovering, move_sec=10.0, collect_sec=50.0)
    expected_move = 10.0 * (257.0 * 10.0 + 326.0 * 50.0) / 3600.0 / 125.0
    expected_hover = 10.0 * 326.0 * 60.0 / 3600.0 / 125.0
    np.testing.assert_allclose(e_move, expected_move, rtol=1e-7)
    np.testing.assert_allclose(e_hover, expected_hover, rtol=1e-7)


def test_rollout_records_dead_agent_mask():
    r = EpisodeRollout()
    r.add(
        np.zeros((2, 3), np.float32), np.zeros(4, np.float32),
        np.zeros(2), np.zeros(2), 1.0, True, 0.0,
        np.array([True, False]),
    )
    out = r.finish(last_value=0.0, gamma=0.99, gae_lambda=0.95)
    assert out["agent_active"].tolist() == [[True, False]]


def test_serving_trajectory_is_saved_per_upper_block():
    env = make_env(n_uav=4, episode_len=4)
    env.step(np.ones(4, np.int64), deterministic_low=True, collect_low_replay=False)
    assert len(env.history["trajectory_blocks"]) == 1
    tracks = env.history["trajectory_blocks"][0]
    assert sorted(tracks) == [0, 1, 2, 3]
    assert all(traj.shape == (2, 2) for traj in tracks.values())


def test_safe_random_low_warmup_stays_inside_world():
    env = make_env(n_uav=4, episode_len=4)
    env.cfg.low_random_steps = 100
    env.step(np.ones(4, np.int64), deterministic_low=False, collect_low_replay=True)
    assert env.history["low_oob_count"][-1] == 0


def test_upper_reward_includes_user_waiting_term_and_weights():
    env = make_env(n_uav=4, episode_len=4)
    env.cfg.serving_weight = 3.0
    env.cfg.waiting_weight = 1.0
    for u, status in zip(env.uavs, [0, 0, 1, 2]):
        u.status = status
    reward, serving, waiting, buffer, death = env._calc_reward(0.0, 0)
    np.testing.assert_allclose(serving, 3.0 * 2.0 / 4.0)
    np.testing.assert_allclose(waiting, 1.0 * 1.0 / 4.0)
    np.testing.assert_allclose([buffer, death], [0.0, 0.0])
    np.testing.assert_allclose(reward, serving + waiting)
