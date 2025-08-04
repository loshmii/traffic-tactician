"""
Generate an animated GIF rollout of a Gym-registered Grid6x6Env.

This script uses `gym.make` to create the env with `render_mode="rgb_array"`,
rolls out either a random policy or a provided SB3 PPO policy for a fixed
number of steps, and writes the frames to an animated GIF.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import manhattan6x6  # ensure the package is imported for registration


def make_gif(
    env_id: str,
    model,
    output_path: Path,
    steps: int = 2000,
    fps: int = 15,
    seed: int | None = None,
) -> None:
    """
    Roll out the given model (or random policy if model is None) in the
    environment created by `gym.make(env_id, render_mode="rgb_array")`.
    Capture up to `steps` timesteps, render each frame, and save as a GIF at
    `output_path`.

    Args:
        env_id: Gym ID of the environment to create (must support "rgb_array").
        model: A Stable-Baselines3 model with a `.predict()` method, or None.
        output_path: Path to write the resulting GIF file.
        steps: Maximum number of simulation timesteps.
        fps: Frames per second for the GIF playback.
        seed: Optional random seed for environment reproducibility.
    """
    # Build the env with rendering enabled
    env = gym.make(env_id, render_mode="rgb_array")
    if seed is not None:
        env.reset(seed=seed)

    # Prepare GIF writer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = 1 / fps
    obs, _ = env.reset()
    with imageio.get_writer(output_path, mode="I", duration=duration) as writer:
        for _ in range(steps):
            # Choose action: random or deterministic from model
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, _, done, truncated, _ = env.step(action)
            frame = env.render()
            writer.append_data(np.asarray(frame))

            if done or truncated:
                break


def main() -> None:
    p = argparse.ArgumentParser(
        description="Create an animated GIF rollout of a Grid6x6Env via Gym."
    )
    p.add_argument(
        "--env-id",
        type=str,
        default="Manhattan6x6-v0",
        help="Gym ID of the environment to roll out (must be registered).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save the output GIF.",
    )
    p.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="Path to a saved Stable-Baselines3 PPO model (.zip), if you want to use it.",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Maximum number of timesteps to simulate.",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Frames per second for the GIF playback.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    args = p.parse_args()

    # Load the SB3 model if given
    model = None
    if args.policy:
        from stable_baselines3 import PPO

        model = PPO.load(str(args.policy))

    # Generate the GIF
    make_gif(
        env_id=args.env_id,
        model=model,
        output_path=args.output,
        steps=args.steps,
        fps=args.fps,
        seed=args.seed,
    )


__all__ = ["make_gif"]

if __name__ == "__main__":
    main()
