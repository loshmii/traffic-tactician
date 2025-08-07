import os, sys, numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from manhattan6x6 import Manhattan6x6Env
from highway_env.envs import HighwayEnvFast


def test_reset():
    obs, _ = Manhattan6x6Env().reset(seed=0)
    assert obs.shape == (19,) and obs.dtype == np.float32


def test_step():
    env = Manhattan6x6Env()
    obs, _ = env.reset(seed=0)
    obs, r, term, trunc, _ = env.step(0)
    assert obs.shape == (19,) and np.isfinite(r)
    assert isinstance(term, (bool, np.bool_)) and isinstance(trunc, (bool, np.bool_))


def test_reward():
    env = Manhattan6x6Env()
    env.reset(seed=0)
    env._last_acc = -100  # induce big jerk
    _, r, _, _, _ = env.step(0)
    assert r < -1.0


def test_goal():
    env = Manhattan6x6Env(config={"spawn_vehicles": 0})
    obs, _ = env.reset(seed=0)
    tot, step, done, trunc = 0.0, 0, False, False
    while not (done or trunc) and step < 2000:
        act = 1 if step < 20 else 0  # accelerate then cruise
        obs, r, done, trunc, _ = env.step(act)
        tot += r
        step += 1
    assert done and tot >= 80