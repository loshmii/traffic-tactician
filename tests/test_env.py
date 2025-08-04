import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import networkx as nx

from manhattan6x6.env import Manhattan6x6Env

def test_reset():
    env = Manhattan6x6Env()
    obs, _ = env.reset()
    assert obs.shape == (64,)

def test_step():
    env = Manhattan6x6Env()
    obs, _ = env.reset()
    obs, rew, term, trunc, info = env.step(env.action_space.sample())
    assert isinstance(obs, np.ndarray)
    assert isinstance(rew, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    assert isinstance(info, dict)

def shortest_path_driver(env: Manhattan6x6Env):
    path = nx.shortest_path(env.graph, source=env.start, target=env.goal)
    headings = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    total_reward = 0.0
    obs, _ = env.reset(seed=42)
    for target in path[1:]:
        diff = (target[0] - env.position[0], target[1] - env.position[1])
        desired = headings.index(diff)
        if desired == env.heading:
            action = 0
        elif (desired - env.heading) % 4 == 1:
            action = 3
        else:
            action = 4
        obs, rew, term, trunc, _ = env.step(action)
        total_reward += rew
        if term or trunc:
            break
    return total_reward, term, trunc

def test_solve():
    env = Manhattan6x6Env()
    total_reward, term, trunc = shortest_path_driver(env)
    assert term and not trunc
    assert total_reward >= 50
