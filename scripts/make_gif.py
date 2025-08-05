"""Generate an animated GIF rollout of a Manhattan grid‑driving environment.

The script creates the env via ``gym.make`` with ``render_mode='rgb_array'``,
then steps either a random policy or a supplied Stable‑Baselines3 PPO model for
``--steps`` timesteps and writes the rendered frames to a GIF.

Usage example
-------------
$ python scripts/make_gif.py --output runs/before.gif --steps 2000 \
      --env-id Manhattan6x6-v0 --seed 42 --no-traffic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import manhattan6x6  # noqa: F401 – registers the envs via import side‑effects


def make_gif(
    env_id: str,
    model,
    output_path: Path,
    env_config: dict | None = None,
    steps: int = 2_000,
    fps: int = 15,
    seed: int | None = None,
) -> None:
    """Roll out *model* (or a random policy) in *env_id* and save a GIF.*"""

    # ------------------------------------------------------------------
    # Environment -------------------------------------------------------
    # ------------------------------------------------------------------
    # Highway‑Env expects custom parameters to be passed under the ``config``
    # kwarg, *not* as arbitrary kwargs.  We therefore forward ``env_config``
    # this way so callers can tweak things like the number of background cars
    # (e.g. ``{"spawn_vehicles": 0}``).
    env = gym.make(env_id, render_mode="rgb_array", config=env_config or {})

    # Seed *after* creation so that the RNGs inside Highway‑Env pick it up.
    obs, _ = env.reset(seed=seed)

    # ------------------------------------------------------------------
    # GIF writer --------------------------------------------------------
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = 1 / fps
    with imageio.get_writer(output_path, mode="I", duration=duration) as writer:
        terminated = truncated = False
        for _ in range(steps):
            # Choose an action ------------------------------------------------
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)

            # Environment step ----------------------------------------------
            obs, _, terminated, truncated, _ = env.step(action)
            frame = env.render()
            writer.append_data(np.asarray(frame))

            if terminated or truncated:
                break

    env.close()


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create an animated GIF rollout of a Manhattan grid traffic env.",
    )
    p.add_argument(
        "--env-id",
        type=str,
        default="Manhattan6x6-v0",
        help="Gymnasium ID of the registered environment.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to save the resulting GIF.",
    )
    p.add_argument(
        "--policy",
        type=Path,
        help="Path to a saved Stable‑Baselines3 PPO model (.zip). If omitted, a random policy is used.",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=2_000,
        help="Maximum number of environment steps to record.",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Playback frames per second for the resulting GIF.",
    )
    p.add_argument("--seed", type=int, help="Optional RNG seed for reproducibility.")

    # Convenience flag: completely disable background traffic
    p.add_argument(
        "--no-traffic",
        action="store_true",
        help="Spawn the environment with zero background vehicles.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Optionally load SB3 model -------------------------------------------
    model = None
    if args.policy:
        from stable_baselines3 import PPO

        model = PPO.load(str(args.policy))

    # Compose env config ---------------------------------------------------
    env_cfg = {}
    if args.no_traffic:
        env_cfg["spawn_vehicles"] = 0

    # Generate the GIF -----------------------------------------------------
    make_gif(
        env_id=args.env_id,
        model=model,
        output_path=args.output,
        env_config=env_cfg,
        steps=args.steps,
        fps=args.fps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
