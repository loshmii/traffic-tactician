import imageio
from stable_baselines3 import PPO

from manhattan6x6.env import Manhattan6x6Env


def main():
    env = Manhattan6x6Env()
    model = PPO.load("policies/ppo_manhattan")
    frames = []
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        frames.append(env.render())
        done = terminated or truncated
    imageio.mimsave("demo.gif", frames, fps=15)
    env.close()


if __name__ == "__main__":
    main()
