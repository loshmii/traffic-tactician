from __future__ import annotations
"""Refactored PPO trainer for the 2×2 Manhattan grid with curriculum phases.

Key features:
- Strict "continue within a phase" top-up mode (with optional bootstrap).
- Optional no-save mode.
- Deterministic run folder via --run-name.
- Episodic logging + EMA plotting unchanged.
- NEW: --save-every (episodic checkpoints, single rotating _ckpt) and --eval-every/--eval-episodes (periodic eval).
- NEW: --load-from-model / --load-from-env to hard-override loader.
- Loader prefers newest variant of requested phase (unless --strict-phase-load), prints which files were loaded.

Usage examples are at the bottom of the file (if __name__ == "__main__").
"""

import argparse
import csv
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

import manhattan6x6  # registers the envs
from scripts.make_gif import make_gif

GYM_ENV_ID = "Manhattan2x2-v0"


# =============================
# Logging & utility callbacks
# =============================
class RewardLogger(BaseCallback):
    """Log episode returns to CSV and optionally show tqdm progress."""
    def __init__(self, total_episodes: int, csv_path: Path, use_tqdm: bool = True):
        super().__init__()
        self.total_episodes = total_episodes
        self.csv_path = csv_path
        self.use_tqdm = use_tqdm
        self.comp_keys = None
        self._header_written = False

    @staticmethod
    def _scalar(x) -> float:
        if x is None:
            return 0.0
        if isinstance(x, (int, float, np.floating)):
            return float(x)
        arr = np.asarray(x)
        if arr.ndim == 0:
            return float(arr)
        if arr.size == 1:
            return float(arr.ravel()[0])
        # For vector metrics (e.g., shape components), log the sum; adjust if you prefer mean/max
        return float(arr.sum())

    def _open(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if not self._header_written else "a"
        self._f = self.csv_path.open(mode, newline="")
        self._writer = csv.writer(self._f)

    def _on_training_start(self) -> None:
        self._open()
        if self.use_tqdm:
            from tqdm import tqdm
            self.pbar = tqdm(total=self.total_episodes, desc="Episodes")
        self.episode_idx = 0

    def _on_step(self) -> bool:
        # Grab infos (VecEnv -> list), find the first one that has an "episode" entry
        infos = self.locals.get("infos", [])
        info = None
        if isinstance(infos, (list, tuple)):
            for d in infos:
                if isinstance(d, dict) and "episode" in d:
                    info = d
                    break
        elif isinstance(infos, dict) and "episode" in infos:
            info = infos

        if info is None:
            return True  # nothing to log this step

        ep_mon = info.get("episode", {})
        total_reward = ep_mon.get("total_reward", ep_mon.get("r", 0.0))
        length = ep_mon.get("length", ep_mon.get("l", 0.0))

        # Merge other info fields (from VecMonitor info_keywords)
        ep_stats = {k: v for k, v in info.items() if k != "episode"}
        ep_stats.update({"total_reward": total_reward, "length": length})

        if self.comp_keys is None:
            ordered = ["total_reward", "length"]
            others = sorted(k for k in ep_stats.keys() if k not in ordered)
            self.comp_keys = ordered + others
            self._writer.writerow(["episode", *self.comp_keys])
            self._header_written = True

        row = [self.episode_idx] + [self._scalar(ep_stats.get(k, 0.0)) for k in self.comp_keys]
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
    """Tracks EMAs, logs to TensorBoard, and mirrors to a CSV."""
    def __init__(self, span: int = 100, verbose: int = 0, csv_path: Path | None = None):
        super().__init__(verbose)
        self.span = span
        self.alpha = 2 / (span + 1)
        self.ema = {}
        self.step_count = 0
        self.csv_path = csv_path
        self._csv_header_written = False
        self._csv_file = None
        self._csv = None

    def _open_csv(self, keys):
        if self.csv_path is None:
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if not self._csv_header_written else "a"
        self._csv_file = self.csv_path.open(mode, newline="")
        self._csv = csv.writer(self._csv_file)
        if not self._csv_header_written:
            self._csv.writerow(["episode", *keys])
            self._csv_header_written = True

    def _close_csv(self):
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv = None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if not done or "episode" not in info:
                continue
            ep_raw = info["episode"]
            ep = {k: float(v) for k, v in ep_raw.items() if k not in ("l", "r")}
            if not self.ema:
                for k, v in ep.items():
                    self.ema[k] = float(v)
                # CSV setup
                self._open_csv(sorted(self.ema.keys()))
            # update
            for k, v in ep.items():
                self.ema[k] = self.alpha * float(v) + (1 - self.alpha) * self.ema[k]
                self.logger.record(f"ema/{k}", self.ema[k])
            # also keep standard summaries
            self.logger.record("rollout/ep_rew_mean", float(ep.get("total_reward", 0.0)))
            self.logger.record("rollout/ep_len_mean", float(ep.get("length", 0.0)))
            # bump episode counter
            self.step_count += 1
            self.logger.record("train/episode", self.step_count)
            # mirror to CSV
            if self._csv is not None:
                row = [self.step_count] + [self.ema[k] for k in sorted(self.ema.keys())]
                self._csv.writerow(row)
        return True

    def _on_training_end(self) -> None:
        self._close_csv()


class EarlySuccessStop(BaseCallback):
    """Stop training after successive goal-reaching episodes."""
    def __init__(self, patience: int = 50, verbose: int = 0):
        super().__init__(verbose)
        self.patience = patience
        self._streak = 0

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if float(info.get("goal", 0.0)) > 0.0:
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


class PeriodicCheckpointAndEval(BaseCallback):
    """Every N episodes: overwrite a single rotating checkpoint ("_ckpt") and/or run eval."""
    def __init__(
        self,
        phase_index: int,
        save_every: int | None,
        eval_every: int | None,
        eval_episodes: int,
        save_fn,
        eval_fn,
        dont_save: bool,
    ):
        super().__init__()
        self.phase_index = phase_index
        self.save_every = save_every if (save_every and save_every > 0) else None
        self.eval_every = eval_every if (eval_every and eval_every > 0) else None
        self.eval_episodes = max(1, eval_episodes)
        self.save_fn = save_fn
        self.eval_fn = eval_fn
        self.dont_save = dont_save
        self._episodes = 0

    def _on_step(self) -> bool:
        if not self.locals.get("dones", [False])[0]:
            return True
        self._episodes += 1

        if self.save_every and not self.dont_save and (self._episodes % self.save_every == 0):
            self.save_fn(self.phase_index, suffix="_ckpt")

        if self.eval_every and (self._episodes % self.eval_every == 0):
            self.eval_fn(self.eval_episodes)
        return True


# =============================
# Plotting helpers
# =============================

def stats_from_csv(csv_path: Path, span: int = 100):
    df = pd.read_csv(csv_path)
    df = df.drop(columns="episode", errors="ignore")
    ema = df.ewm(span=span, adjust=False).mean()
    last_ema = ema.iloc[-1]
    print(f"Exponential Moving Averages (span={span}) at final episode from {csv_path}:")
    for col in df.columns:
        print(f"{col:20s} EMA = {last_ema[col]:8.3f}")


def plot_csv(csv_path: Path, png_path: Path, span: int = 100) -> None:
    df = pd.read_csv(csv_path)
    df_ema = df.set_index("episode").ewm(span=span, adjust=False).mean().reset_index()
    ax = df_ema.plot(
        x="episode",
        y=[c for c in df_ema.columns if c not in ["episode", "l", "r", "t"]],
        figsize=(10, 6),
        grid=True,
        title=f"EMA (span={span}) of Reward Components",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("EMA value")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()


# =============================
# CLI & config
# =============================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on the 2×2 Manhattan grid with curriculum phases.")
    p.add_argument("--episodes-per-phase", type=int, nargs='+', required=True,
                  help="Episode counts per phase (length must match number of phases).")
    p.add_argument("--continue-in-phase", type=int, default=None,
                  help="1-based phase index to continue training within (strict existence check).")
    p.add_argument("--extra-episodes", type=int, default=None,
                  help="Additional episodes to run in the selected phase; requires --continue-in-phase.")
    p.add_argument("--bootstrap-if-missing", action="store_true",
                  help="If continuing but phase assets are missing, start fresh instead of erroring.")
    p.add_argument("--dont-save", action="store_true", help="Skip saving model/env/checkpoints.")
    p.add_argument("--run-name", type=str, default=None, help="Folder name under runs/. If omitted, timestamp is used.")

    # Loader behavior
    p.add_argument("--strict-phase-load", action="store_true",
                   help="Require exact phase checkpoint names (no variant discovery).")
    p.add_argument("--load-from-model", type=str, default=None,
                   help="Absolute/relative path to a specific .zip model checkpoint to load.")
    p.add_argument("--load-from-env", type=str, default=None,
                   help="Absolute/relative path to a specific VecNormalize env stats directory to load.")

    # QoL
    p.add_argument("--no-gif", action="store_true", help="Skip GIF generation.")
    p.add_argument("--no-tqdm", action="store_true", help="Hide tqdm progress bar.")
    p.add_argument("--no-early-stop", action="store_true", help="Disable early stopping criterion.")
    p.add_argument("--steps", type=int, default=2000, help="Max timesteps per episode (env limit).")
    p.add_argument("--fps", type=int, default=15, help="FPS for GIFs.")
    p.add_argument("--seed", type=int, default=None, help="Random seed.")

    # NEW
    p.add_argument("--save-every", type=int, default=0,
                  help="Checkpoint every N episodes within a phase (0 disables). Ignored if --dont-save.")
    p.add_argument("--eval-every", type=int, default=0,
                  help="Run deterministic evaluation every N episodes (0 disables).")
    p.add_argument("--eval-episodes", type=int, default=10,
                  help="Number of episodes to use at each eval.")

    return p.parse_args()


def linear_schedule(initial_value: float):
    def f(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return f


# =============================
# Phase I/O and factories
# =============================

def make_env() -> gym.Env:
    return gym.make(GYM_ENV_ID)


def make_vec_env(log_path: Path) -> VecNormalize:
    raw_env = DummyVecEnv([make_env])
    venv = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    venv = VecMonitor(
        venv,
        filename=str(log_path / "monitor.csv"),
        info_keywords=(
            "total_reward", "shape", "ttc", "step_cost", "jerk",
            "crash", "goal", "off_road_hard_event", "off_road_soft_penalty",
        ),
    )
    return venv


def make_fresh_model(env: VecNormalize) -> PPO:
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        clip_range=0.2,
        n_steps=2048,
        batch_size=512,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log="logs/ppo_manhattan2x2",
    )


def save_env(env: VecNormalize, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    env.save(path)


def save_model(model: PPO, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def load_env(path: Path, venv_template: VecNormalize) -> VecNormalize:
    return VecNormalize.load(path, venv_template)


def load_model(path: Path, env: VecNormalize) -> PPO:
    return PPO.load(path, env=env)


@dataclass
class PhasePaths:
    model: Path
    env: Path


def phase_paths(runs_dir: Path, i0: int) -> PhasePaths:
    """i0 is 0-based phase index. Return canonical base paths."""
    return PhasePaths(
        model=runs_dir / "model" / f"ppo_phase_{i0+1}.zip",
        env=runs_dir / "env" / f"train_env_phase_{i0+1}",
    )


def phase_assets_exist(paths: PhasePaths) -> bool:
    return paths.model.exists() and paths.env.exists()


def _latest_variant_for_phase(runs_dir: Path, i0: int) -> tuple[Path | None, Path | None]:
    """Find the newest model/env pair for phase i0 among variants like
    ppo_phase_<k>.zip, ppo_phase_<k>_ckpt.zip, ppo_phase_<k>* and matching env dirs.
    Returns (model_path, env_path) or (None, None) if not found.
    """
    import glob, os
    k = i0 + 1
    model_candidates = []
    for p in glob.glob(str(runs_dir / "model" / f"ppo_phase_{k}*.zip")):
        try:
            model_candidates.append((os.path.getmtime(p), Path(p)))
        except FileNotFoundError:
            pass
    if not model_candidates:
        return None, None
    # pick newest model
    model_candidates.sort(key=lambda x: x[0], reverse=True)
    mpath = model_candidates[0][1]

    # try exact-matching env dir name first; else newest variant
    env_base = runs_dir / "env"
    variants = [env_base / f"train_env_phase_{k}"]
    variants += [Path(p) for p in glob.glob(str(env_base / f"train_env_phase_{k}*"))]
    variants = [v for v in variants if v.exists()]
    if variants:
        variants.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return mpath, variants[0]
    return mpath, None


# =============================
# Main
# =============================

def main() -> None:
    args = _parse_args()

    # Curriculum (phase-specific env kwargs merged cumulatively inside the env via set_curriculum)
    phases: list[dict] = [
        {"spawn_vehicles": 0},
        {"shaping_coef": 1.0, "step_cost": 0.05},
        {"spawn_vehicles": 5,  "shaping_coef": 0.5},
        {"spawn_vehicles": 20, "shaping_coef": 0.2},
        {"shaping_coef": 0.0},
    ]

    episodes_per_phase = list(args.episodes_per_phase)
    if len(episodes_per_phase) != len(phases):
        print(
            f"Error: --episodes-per-phase length ({len(episodes_per_phase)}) does not match number of phases ({len(phases)}).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Continue-in-phase override (strict by default)
    if args.continue_in_phase is not None:
        if args.extra_episodes is None:
            print("Error: --continue-in-phase requires --extra-episodes.", file=sys.stderr)
            sys.exit(1)
        if not (1 <= args.continue_in_phase <= len(phases)):
            print(f"Error: --continue-in-phase must be in [1, {len(phases)}].", file=sys.stderr)
            sys.exit(1)
        idx = args.continue_in_phase - 1
        episodes_per_phase = [0] * len(phases)
        episodes_per_phase[idx] = int(args.extra_episodes)

    runs = Path("runs"); runs.mkdir(exist_ok=True)
    run_name = args.run_name if args.run_name else datetime.now().strftime("%b-%d-%H-%M")
    log_dir = runs / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    total_episodes = sum(episodes_per_phase)

    # Build base env/model (may be overwritten by loaders per phase)
    train_env = make_vec_env(log_dir)
    model = make_fresh_model(train_env)

    # Logger and optional early stop
    cb_logger = RewardLogger(total_episodes, log_dir / "reward_log.csv", use_tqdm=not args.no_tqdm)
    cb_ema = EMAPlottingCallback(span=100, verbose=1, csv_path=log_dir / "ema_log.csv")
    early_stop = None if args.no_early_stop else EarlySuccessStop(patience=50, verbose=1)

    last_cfg: dict | None = None

    def set_phase_cfg(env: VecNormalize, cfg: dict):
        # Relay to underlying envs
        env.env_method("set_curriculum", **cfg)

    def strict_load_for_phase(i0: int) -> tuple[PPO, VecNormalize]:
        """Load checkpoint for training phase i0.
        Priority:
          1. If --load-from-model and --load-from-env are set, load those exactly.
          2. If --strict-phase-load: require exact canonical names (ppo_phase_k.zip, train_env_phase_k).
          3. Else: use the **newest variant** for that phase (matches * and _ckpt) if available; if none, fall back to newest overall phase.
          4. If nothing exists: error in continue mode (unless --bootstrap-if-missing), else start fresh.
        """
        nonlocal train_env, model
        cont_mode = args.continue_in_phase is not None

        def _attach(env_path: Path, model_path: Path):
            nonlocal train_env, model
            train_env = load_env(env_path, train_env)
            train_env.training = True
            model = load_model(model_path, train_env)
            print(f"[LOAD] model={model_path} env={env_path}")
            return model, train_env

        # 1) Absolute overrides must be both provided
        if (args.load_from_model is not None) or (args.load_from_env is not None):
            if not (args.load_from_model and args.load_from_env):
                print("Error: provide BOTH --load-from-model and --load-from-env.", file=sys.stderr)
                sys.exit(1)
            mp = Path(args.load_from_model)
            ep = Path(args.load_from_env)
            if not mp.exists() or not ep.exists():
                print("Error: --load-from-model or --load-from-env path does not exist.", file=sys.stderr)
                sys.exit(1)
            return _attach(ep, mp)

        # 2) Strict exact phase
        if args.strict_phase_load:
            paths = phase_paths(runs, i0)
            if phase_assets_exist(paths):
                return _attach(paths.env, paths.model)
        else:
            # 3) Newest variant for requested phase
            mp, ep = _latest_variant_for_phase(runs, i0)
            if mp is not None and ep is not None:
                return _attach(ep, mp)

        # 3b) Non-strict: fallback to newest among all phases
        if not args.strict_phase_load:
            for j in range(len(phases) - 1, -1, -1):
                mp, ep = _latest_variant_for_phase(runs, j)
                if mp is not None and ep is not None:
                    return _attach(ep, mp)

        # 4) Nothing found
        if cont_mode and not args.bootstrap_if_missing:
            p = phase_paths(runs, i0)
            print(
                f"Error: No saved checkpoints found for requested phase {i0+1}."
                f"Looked for {p.model} / {p.env} and variants. Use --bootstrap-if-missing to start fresh.",
                file=sys.stderr,
            )
            sys.exit(1)
        return model, train_env

    def save_after_phase(i0: int, suffix: str = "") -> None:
        if args.dont_save:
            return
        p = phase_paths(runs, i0)
        # Allow checkpoint naming via constant suffix ("_ckpt") to overwrite previous
        model_path = p.model if suffix == "" else p.model.with_name(p.model.stem + suffix).with_suffix(".zip")
        env_path = p.env if suffix == "" else Path(str(p.env) + suffix)
        save_model(model, model_path)
        save_env(train_env, env_path)

    # Lightweight evaluation that does not update normalization stats
    def run_eval(eval_episodes: int) -> None:
        was_training = train_env.training
        train_env.training = False
        returns = []
        lengths = []
        for _ in range(eval_episodes):
            obs = train_env.reset()
            done = False
            ep_ret = 0.0
            ep_len = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = train_env.step(action)
                r = float(reward[0]) if isinstance(reward, (list, tuple, np.ndarray)) else float(reward)
                d = bool(done[0]) if isinstance(done, (list, tuple, np.ndarray)) else bool(done)
                ep_ret += r
                ep_len += 1
                if d:
                    break
            returns.append(ep_ret)
            lengths.append(ep_len)
        train_env.training = was_training
        mean_ret = float(np.mean(returns)) if returns else 0.0
        mean_len = float(np.mean(lengths)) if lengths else 0.0
        # TensorBoard: write immediately with current timesteps as step index
        model.logger.record("eval/return_mean", mean_ret)
        model.logger.record("eval/len_mean", mean_len)
        # dump to ensure it appears promptly
        try:
            model.logger.dump(step=int(model.num_timesteps))
        except Exception:
            pass
        # CSV mirror
        eval_csv = log_dir / "eval_log.csv"
        header_needed = not eval_csv.exists()
        with eval_csv.open("a", newline="") as f:
            w = csv.writer(f)
            if header_needed:
                w.writerow(["episode", "return_mean", "len_mean"]) 
            # approximate episode index: use EMA callback's episode counter if available
            ep_idx = getattr(cb_ema, "step_count", 0)
            w.writerow([ep_idx, mean_ret, mean_len])
        print(f"[EVAL] episodes={eval_episodes} mean_return={mean_ret:.3f} mean_len={mean_len:.1f}")

    # ============
    # Train loop
    # ============
    completed = 0
    for i0, (cfg, eps) in enumerate(zip(phases, episodes_per_phase)):
        if eps <= 0:
            continue
        print(f"=== Phase {i0+1}: running {eps} episodes with cfg {cfg} ===")

        # Load/Bootstrap model-env for this phase
        model, train_env = strict_load_for_phase(i0)

        # Apply curriculum config to env
        set_phase_cfg(train_env, cfg)
        last_cfg = cfg

        # Callbacks for this phase
        stop_cb = StopOnEpisodes(eps)
        phase_cbs = [cb_logger, cb_ema, stop_cb]
        if early_stop is not None:
            early_stop._streak = 0
            phase_cbs.append(early_stop)

        periodic = PeriodicCheckpointAndEval(
            phase_index=i0,
            save_every=args.save_every,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            save_fn=save_after_phase,
            eval_fn=run_eval,
            dont_save=args.dont_save,
        )
        phase_cbs.append(periodic)
        callbacks = CallbackList(phase_cbs)

        # Timesteps per episode is bounded by args.steps
        total_timesteps = args.steps * eps
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        completed += eps
        print(f"--- Phase {i0+1} done: cumulative episodes = {completed} ---")

        # Save final snapshot for the phase
        save_after_phase(i0)

    # GIF (pre: disabled; post: last config)
    if not args.no_gif:
        make_gif(
            env_id=GYM_ENV_ID,
            model=model,
            output_path=log_dir / "after.gif",
            env_config=last_cfg or phases[0],
            steps=args.steps,
            fps=args.fps,
            seed=args.seed,
        )

    # Plots & stats
    plot_csv(log_dir / "reward_log.csv", log_dir / "reward_curve.png")
    stats_from_csv(log_dir / "reward_log.csv")

    sys.exit(0)


if __name__ == "__main__":
    main()
