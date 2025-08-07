from __future__ import annotations
"""Train PPO on the 2×2 traffic grid with phased curriculum, log rewards, plot curve, and create GIFs."""
import argparse
import csv
import sys
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.monitor import Monitor

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
        self.comp_keys = None
        self._header_written = False
        
    def _open(self) :
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if not self._header_written else "a"
        self._f = self.csv_path.open(mode, newline="")
        self._writer = csv.writer(self._f)   

    def _on_training_start(self) -> None:
        self._open()
        if self.use_tqdm:
            from tqdm import tqdm
            self.pbar = tqdm(total=self.total_episodes, desc="Episodes")

    def _on_step(self) -> bool:
        if not self.locals["dones"][0]:
            return True
        
        infos = self.locals["infos"][0]
        ep_stats = infos["episode"] if "episode" in infos else infos

        if self.comp_keys is None:
            self.comp_keys = ["total_reward", *sorted(k for k in ep_stats if k != "total_reward")]
            self._writer.writerow(["episode", *self.comp_keys])
            self._header_written = True
        
        if not hasattr(self, "episode_idx"):
            self.episode_idx = 0
        row = [self.episode_idx] + [ep_stats.get(k, 0.0) for k in self.comp_keys]
        self._writer.writerow(row)

        if self.use_tqdm:
            self.pbar.update(1)
        self.episode_idx += 1

        return self.episode_idx < self.total_episodes

    def _on_training_end(self) -> None:
        if self.use_tqdm:
            self.pbar.close()
        self._f.close()

class EMAPlottingCallback(BaseCallback):
    def __init__(self, span: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.span = span
        self.alpha = 2 / (span + 1)  # EMA smoothing factor
        self.ema = {}
        self.last_values = deque()
        self.step_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if not done or "episode" not in info:
                continue
            ep = info["episode"]

            if not self.ema:
                for k,v in ep.items():
                    if k in ("l", "r"):
                        continue
                    self.ema[k] = v
            for k, v in ep.items():
                if k in ("l", "r"):
                    continue
                self.ema[k] = self.alpha * float(v) + (1 - self.alpha) * self.ema[k]
                self.logger.record(f"ema/{k}", self.ema[k])
            self.logger.record("rollout/ep_rew_mean", ep.get("total_reward", 0.0))
            self.logger.record("rollout/ep_len_mean", ep.get("l", 0.0))

            self.step_count += 1
            self.logger.record("train/episode", self.step_count)
    
        return True

class EarlySuccessStop(BaseCallback):
    """Stop training after successive goal-reaching episodes."""

    def __init__(self, patience: int = 50, verbose: int = 0):
        super().__init__(verbose)
        self.patience = patience
        self._streak = 0

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("goal", 0.0) > 0.0:
                self._streak += 1
            else:
                self._streak = 0
        if self._streak >= self.patience:
            if self.verbose:
                print(f"Stopping early after {self._streak} consecutive goal-reaching episodes.")
            return False
        return True


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

def stats_from_csv(csv_path: Path, span: int = 100):
    """Compute and display EMAs of each reward component at the last episode."""
    df = pd.read_csv(csv_path)
    df = df.drop(columns="episode", errors="ignore")
    #new: compute EMA for each column
    ema = df.ewm(span=span, adjust=False).mean()
    #new: get last EMA values
    last_ema = ema.iloc[-1]
    print(f"Exponential Moving Averages (span={span}) at final episode from {csv_path}:")
    for col in df.columns:
        print(f"{col:20s} EMA = {last_ema[col]:8.3f}")


def plot_csv(csv_path: Path, png_path: Path, span: int = 100) -> None:
    """Plot EMAs of all reward components over episodes and save to PNG."""
    df = pd.read_csv(csv_path)
    #new: compute EMA series and include episode index
    df_ema = df.set_index("episode").ewm(span=span, adjust=False).mean().reset_index()
    #new: plot EMA of each component
    ax = df_ema.plot(
        x="episode",
        y=[col for col in df_ema.columns if col not in ["episode", "l", "r", "t"]],
        figsize=(10, 6),
        grid=True,
        title=f"EMA (span={span}) of Reward Components",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("EMA value")
    # save figure
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
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
        {"spawn_vehicles": 0,},
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
    """if not args.no_gif:
        make_gif(
            env_id=GYM_ENV_ID,
            model=None,
            output_path=runs / "before.gif",
            env_config=phases[0],
            steps=args.steps,
            fps=args.fps,
            seed=args.seed,
        )"""

    def make_env() -> gym.Env:
        return gym.make(GYM_ENV_ID)

    raw_env = DummyVecEnv([make_env])
    train_env = VecNormalize(
        raw_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )
    train_env = VecMonitor(
        train_env,
        filename=str(runs / "monitor.csv"),
        info_keywords=("total_reward", "shape", "ttc", "step_cost", "jerk",
            "crash", "goal", "off_road_hard_event", "off_road_soft_penalty"),
    )

    model = PPO(
        "MlpPolicy", train_env,
        learning_rate=2.5e-4,
        n_steps=4096,
        batch_size=4096,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log="logs/ppo_manhattan2x2"
    )

    cb_logger = RewardLogger(total_episodes, runs / "reward_log.csv", not args.no_tqdm)
    cb_ema = EMAPlottingCallback(span=100, verbose=1)
    early_stop = EarlySuccessStop(patience=50, verbose=1)
    completed = 0
    last_cfg: dict | None = None

    # Phase-wise training
    for idx, (cfg, eps) in enumerate(zip(phases, episodes_per_phase), start=1):
        print(f"=== Phase {idx}: running {eps} episodes with config {cfg} ===")
        if eps <= 0:
            print(f"--- Skipping phase {idx}: zero episodes requested ---")
            continue
        # Apply curriculum and record last config
        if (idx != 1) :
            model.load(runs / "model" / f"ppo_phase_{idx-1}")
        train_env.env_method("set_curriculum", **cfg)
        last_cfg = cfg
        stop_cb = StopOnEpisodes(eps)
        callbacks = CallbackList([cb_logger, stop_cb, early_stop, cb_ema])
        model.learn(
            total_timesteps=int(1e9),
            callback=callbacks,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        completed += eps
        print(f"--- Phase {idx} done: total episodes completed = {completed} ---")
        model.save(runs / "model" / f"ppo_phase_{idx}")
        train_env.save(runs / "env" / f"train_env_phase_{idx}")
    

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
    stats_from_csv(runs / "reward_log.csv")
    sys.exit(0)


if __name__ == "__main__":
    main()
