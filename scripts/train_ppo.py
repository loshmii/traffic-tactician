import os
from functools import partial

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

from manhattan6x6.env import Manhattan6x6Env


class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps: int):
        super().__init__()
        self.progress = tqdm(total=total_timesteps)

    def _on_step(self) -> bool:
        self.progress.update(self.model.n_envs)
        return True

    def _on_training_end(self) -> None:
        self.progress.close()


def make_env(seed: int | None = None):
    def _init():
        env = Manhattan6x6Env()
        if seed is not None:
            env.reset(seed=seed)
        return env

    return _init


def main():
    total_steps = 200_000
    n_envs = 8
    env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=64,
        batch_size=256,
        learning_rate=3e-4,
        verbose=0,
    )
    callback = TqdmCallback(total_steps)
    model.learn(total_timesteps=total_steps, callback=callback)
    os.makedirs("policies", exist_ok=True)
    model.save("policies/ppo_manhattan")
    env.close()


if __name__ == "__main__":
    main()
