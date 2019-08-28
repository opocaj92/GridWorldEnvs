from gym.envs.registration import register

register(
    id='GridWorld-v0',
    entry_point='gym_gridworld.envs:GridWorld',
)
register(
    id='PursuersEvaders-v0',
    entry_point='gym_gridworld.envs:PursuersEvaders',
)