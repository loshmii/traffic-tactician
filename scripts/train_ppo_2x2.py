from __future__ import annotations
"""Train PPO on the 2×2 traffic grid with phased curriculum, log rewards, plot curve, and create GIFs."""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

import manhattan6x6  # registers the envs
from scripts.make_gif import make_gif

GYM_ENV_ID = "Manhattan2x2-v0"


class RewardLogger(BaseCallback):
    """Log episode returns to CSV and optionally show tqdm progress."""
    def __init__(self, total_episodes: int, csv_path: Path, use_tqdm: bool = True):
        super().__init__()
        self.total_episodes = total_episodes
        self.csv_path = csv_path
        self.use_tqdm = use_tqdm
        self.episodes_logged = 0
        self.episode_return = 0.0
        self.pbar = None
        self._header_written = False

    def _on_training_start(self) -> None:
        self.csv_path.parent.mkdir(exist_ok=True)
        mode = "w" if not self._header_written else "a"
        self.file = self.csv_path.open(mode, newline="")
        self.writer = csv.writer(self.file)
        if not self._header_written:
            self.writer.writerow(["episode", "reward"])
            self._header_written = True
        if self.use_tqdm:
            from tqdm import tqdm
            self.pbar = tqdm(total=self.total_episodes)
            if self.episodes_logged > 0:
                self.pbar.update(self.episodes_logged)

    def _on_step(self) -> bool:
        self.episode_return += float(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            self.writer.writerow([self.episodes_logged, self.episode_return])
            if self.pbar:
                self.pbar.update(1)
            self.episodes_logged += 1
            self.episode_return = 0.0
        return self.episodes_logged < self.total_episodes

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()
        self.file.close()


class StopOnEpisodes(BaseCallback):
    """Stop training after a fixed number of episodes."""
    def __init__(self, max_episodes: int):
        super().__init__()
        self.max_episodes = max_episodes
        self.episodes = 0

    def _on_step(self) -> bool:
        if self.locals.get("dones", [False])[0]:
            self.episodes += 1
            if self.episodes >= self.max_episodes:
                return False
        return True


def plot_csv(csv_path: Path, png_path: Path) -> None:
    episodes, rewards = np.loadtxt(csv_path, delimiter=",", skiprows=1, unpack=True)
    plt.plot(episodes, rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    png_path.parent.mkdir(exist_ok=True)
    plt.savefig(png_path)
    plt.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO on the 2×2 Manhattan grid with curriculum phases."
    )
    parser.add_argument(
        "--episodes-per-phase",
        type=int,
        nargs='+',
        required=True,
        help="List of episode counts for each curriculum phase (length must match number of phases)."
    )
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF generation.")
    parser.add_argument("--no-tqdm", action="store_true", help="Hide tqdm progress bar.")
    parser.add_argument("--steps", type=int, default=2000, help="Max timesteps per episode (env limit).")
    parser.add_argument("--fps", type=int, default=15, help="FPS for GIFs.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Define curriculum phases with config updates
    phases: list[dict] = [
        {"spawn_vehicles": 0,  "shaping_coef": 1.0, "step_cost": 0.0},
        {"shaping_coef": 1.0, "step_cost": 0.05},
        {"spawn_vehicles": 5,  "shaping_coef": 0.5},
        {"spawn_vehicles": 20, "shaping_coef": 0.2},
        {"shaping_coef": 0.0},
    ]

    episodes_per_phase = args.episodes_per_phase
    if len(episodes_per_phase) != len(phases):
        print(f"Error: --episodes-per-phase length ({len(episodes_per_phase)}) does not match number of phases ({len(phases)}).", file=sys.stderr)
        sys.exit(1)

    total_episodes = sum(episodes_per_phase)
    runs = Path("runs")
    runs.mkdir(exist_ok=True)

    # Generate "before training" GIF using first phase
    if not args.no_gif:
        make_gif(
            env_id=GYM_ENV_ID,
            model=None,
            output_path=runs / "before.gif",
            env_config=phases[0],
            steps=args.steps,
            fps=args.fps,
            seed=args.seed,
        )

    def make_env() -> gym.Env:
        return gym.make(GYM_ENV_ID)

    train_env = DummyVecEnv([make_env])
    train_env = VecMonitor(train_env)

    model = PPO(
        "MlpPolicy", train_env,
        learning_rate=3e-4,
        gamma=0.995,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=0,
        tensorboard_log="logs/ppo_manhattan2x2"
    )

    cb_logger = RewardLogger(total_episodes, runs / "reward_log.csv", not args.no_tqdm)
    completed = 0
    last_cfg: dict | None = None

    # Phase-wise training
    for idx, (cfg, eps) in enumerate(zip(phases, episodes_per_phase), start=1):
        print(f"=== Phase {idx}: running {eps} episodes with config {cfg} ===")
        if eps <= 0:
            print(f"--- Skipping phase {idx}: zero episodes requested ---")
            continue
        # Apply curriculum and record last config
        train_env.env_method("set_curriculum", **cfg)
        last_cfg = cfg
        stop_cb = StopOnEpisodes(eps)
        callbacks = CallbackList([cb_logger, stop_cb])
        model.learn(
            total_timesteps=int(1e9),
            callback=callbacks,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        completed += eps
        print(f"--- Phase {idx} done: total episodes completed = {completed} ---")

    # Generate "after training" GIF using last applied config
    if not args.no_gif:
        make_gif(
            env_id=GYM_ENV_ID,
            model=model,
            output_path=runs / "after.gif",
            env_config=last_cfg or phases[0],
            steps=args.steps,
            fps=args.fps,
            seed=args.seed,
        )

    plot_csv(runs / "reward_log.csv", runs / "reward_curve.png")
    sys.exit(0)


if __name__ == "__main__":
    main()
