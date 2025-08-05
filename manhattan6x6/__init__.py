from .env import Manhattan6x6Env
from gymnasium.envs.registration import register

register(
    id="Manhattan6x6-v0",
    entry_point="manhattan6x6.env:Manhattan6x6Env",
)
register(
    id="Manhattan2x2-v0",
    entry_point="manhattan6x6.env:Manhattan2x2Env",
)