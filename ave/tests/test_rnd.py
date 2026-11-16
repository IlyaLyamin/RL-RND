"""Unit-тесты RND (Путь B)."""

import numpy as np
import pytest

from src.rnd import RNDModule, RunningMeanStd


def test_running_mean_std_constant_array():
    rms = RunningMeanStd((3, 7, 7))
    const = np.full((3, 7, 7), 5.0)
    for _ in range(10):
        rms.update(const)
    assert np.allclose(rms.mean, 5.0, atol=1e-6)
    assert np.all(rms.var < 1e-6)
    normalized = rms.normalize(const)
    assert np.allclose(normalized, 0.0, atol=1e-4)


def test_running_mean_std_state_roundtrip():
    rms = RunningMeanStd((3, 7, 7))
    rng = np.random.default_rng(0)
    for _ in range(5):
        rms.update(rng.normal(size=(4, 3, 7, 7)))
    state = rms.state_dict()
    clone = RunningMeanStd((3, 7, 7))
    clone.load_state_dict(state)
    assert np.allclose(clone.mean, rms.mean)
    assert np.allclose(clone.var, rms.var)
    assert clone.count == rms.count


def test_intrinsic_reward_decreases_on_repeated_obs():
    rnd = RNDModule(obs_shape=(3, 7, 7), feature_dim=64, device="cpu")
    obs = np.random.randint(0, 10, size=(3, 7, 7)).astype(np.float32)
    rewards = []
    for _ in range(200):
        rewards.append(rnd.intrinsic_reward(obs))
        rnd.update(obs)
    assert rewards[-1] < rewards[0] * 0.5, (rewards[0], rewards[-1])


def test_target_never_updates():
    rnd = RNDModule(obs_shape=(3, 7, 7), feature_dim=16, device="cpu")
    before = [p.detach().clone() for p in rnd.target.parameters()]
    obs = np.zeros((3, 7, 7), dtype=np.float32)
    for _ in range(3):
        rnd.update(obs)
    after = list(rnd.target.parameters())
    for b, a in zip(before, after):
        assert np.allclose(b.cpu().numpy(), a.detach().cpu().numpy())
