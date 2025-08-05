"""Train PPO on the 6×6 traffic grid, log rewards, plot curve, and create GIFs."""

from __future__ import annotations
import matplotlib
matplotlib.use("Agg")  # headless backend

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

import manhattan6x6  # ensure the package is imported for registration
from scripts.make_gif import make_gif  # absolute import

GYM_ENV_ID = "Manhattan6x6-v0"


class RewardLogger(BaseCallback):
    """Write episode returns to CSV (and optional tqdm progress bar)."""

    def __init__(self, episodes: int, csv_path: Path, use_tqdm: bool = True):
        super().__init__()
        self.episodes, self.csv_path, self.use_tqdm = episodes, csv_path, use_tqdm
        self.current, self.count, self.pbar = 0.0, 0, None

    def _on_training_start(self):
        self.csv_path.parent.mkdir(exist_ok=True)
        self.file = self.csv_path.open("w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["episode", "reward"])
        if self.use_tqdm:
            from tqdm import tqdm
            self.pbar = tqdm(total=self.episodes)

    def _on_step(self) -> bool:
        self.current += float(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            self.writer.writerow([self.count, self.current])
            if self.pbar:
                self.pbar.update(1)
            self.current, self.count = 0.0, self.count + 1
        return self.count < self.episodes

    def _on_training_end(self):
        if self.pbar:
            self.pbar.close()
        self.file.close()


def plot_csv(csv_path: Path, png_path: Path):
    episodes, rewards = np.loadtxt(csv_path, delimiter=",", skiprows=1, unpack=True)
    plt.plot(episodes, rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    png_path.parent.mkdir(exist_ok=True)
    plt.savefig(png_path)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--no-tqdm", action="store_true")
    ap.add_argument("--steps", type=int, default=2000,
                    help="Timesteps per GIF rollout")
    ap.add_argument("--fps", type=int, default=15,
                    help="Frames per second for GIFs")
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed for GIF reproducibility")
    args = ap.parse_args()

    # 1) Training env (no rendering)
    train_env_fn = lambda: gym.make(GYM_ENV_ID)
    train_env = DummyVecEnv([train_env_fn])
    train_env = VecMonitor(train_env)

    model = PPO("MlpPolicy",
                train_env,
                learning_rate=3e-4,
                gamma=0.995,
                clip_range=0.2,
                policy_kwargs=dict(net_arch=[256, 256]),
                verbose=0,
                tensorboard_log="logs/ppo_manhattan6x6",
    )
    
    phases = [
        (1_000_000, dict(spawn_vehicles=0, shaping_coef=1.0, step_cost=0.0)),
        (0, dict(shaping_coef=1.0, step_cost=0.05)),
        (0, dict(spawn_vehicles=5, shaping_coef=0.5)),
        (0, dict(spawn_vehicles=20, shaping_coef=0.2)),
        (0, dict(shaping_coef=0.0)),
    ]

    runs = Path("runs")
    runs.mkdir(exist_ok=True)

    # 2) Before-GIF
    if not args.no_gif:
        make_gif(
            env_id=GYM_ENV_ID,
            model=None,
            output_path=runs / "before.gif",
            steps=args.steps,
            fps=args.fps,
            seed=args.seed,
        )

    # 3) Train
    cb = RewardLogger(args.episodes, runs / "reward_log.csv", not args.no_tqdm)
    
    total = 0
    for idx, (steps, kwargs) in enumerate(phases, 1):
        train_env.env_method("set_curriculum", **kwargs)
        model.learn(
            steps,
            callback=cb,
            tb_log_name=f"stage{idx}",
            progress_bar=False,
            reset_num_timesteps=False,
        )
        total += steps
        print(f"phase done - accumulated steps: {total}")

    # 4) After-GIF
    if not args.no_gif:
        make_gif(
            env_id=GYM_ENV_ID,
            model=model,
            output_path=runs / "after.gif",
            steps=args.steps,
            fps=args.fps,
            seed=args.seed,
        )

    # 5) Plot
    plot_csv(runs / "reward_log.csv", runs / "reward_curve.png")
    sys.exit(0)


if __name__ == "__main__":
    main()