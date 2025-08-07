#!/usr/bin/env python3
"""
Generate an animated GIF rollout of a Manhattan grid‑driving environment.

The script uses Gymnasium with `render_mode='rgb_array'` to capture frames
and writes them directly to a GIF file via ImageIO, without displaying any window.

Usage example
-------------
$ python scripts/make_gif.py --output runs/before.gif --steps 2000 \
      --env-id Manhattan6x6-v0 --seed 42 --no-traffic
"""

import argparse
from pathlib import Path
import gymnasium as gym
import imageio


def make_gif(
    env_id: str,
    model,
    output_path: Path,
    env_config: dict | None = None,
    steps: int = 2000,
    fps: int = 15,
    seed: int | None = None,
) -> None:
    """Roll out *model* (or a random policy) in *env_id* and save a GIF."""

    # Create environment with array-based rendering
    env = gym.make(env_id, render_mode="rgb_array", config=env_config or {})
    obs, _ = env.reset(seed=seed)

    # Prepare output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = 1 / fps
    writer = imageio.get_writer(output_path, mode="I", duration=duration)

    # Rollout loop
    terminated = truncated = False
    for _ in range(steps):
        # Select action
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)

        # Step environment
        obs, _, terminated, truncated, _ = env.step(action)
        frame = env.render()  # RGB array, no window shown
        writer.append_data(frame)

        if terminated or truncated:
            break

    # Cleanup
    writer.close()
    env.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create an animated GIF rollout of a Manhattan grid traffic env.",
    )
    p.add_argument("--env-id", type=str, default="Manhattan6x6-v0",
                   help="Gymnasium ID of the registered environment.")
    p.add_argument("--output", type=Path, required=True,
                   help="Where to save the resulting GIF.")
    p.add_argument("--policy", type=Path,
                   help="Path to a saved SB3 PPO model (.zip). If omitted, uses a random policy.")
    p.add_argument("--steps", type=int, default=2000,
                   help="Max number of environment steps to record.")
    p.add_argument("--fps", type=int, default=15,
                   help="Playback frames per second for the GIF.")
    p.add_argument("--seed", type=int,
                   help="Optional RNG seed for reproducibility.")
    p.add_argument("--no-traffic", action="store_true",
                   help="Spawn zero background vehicles.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Load SB3 model if provided
    model = None
    if args.policy:
        from stable_baselines3 import PPO
        model = PPO.load(str(args.policy))

    # Configure environment
    env_cfg = {"spawn_vehicles": 0} if args.no_traffic else {}

    # Create GIF
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
