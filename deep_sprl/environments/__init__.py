from gym.envs.registration import register

register(
    id='ContextualPointMass-v1',
    max_episode_steps=100,
    entry_point='deep_sprl.environments.contextual_point_mass:ContextualPointMass'
)

register(
    id='ContextualPointMass2D-v1',
    max_episode_steps=100,
    entry_point='deep_sprl.environments.contextual_point_mass_2d:ContextualPointMass2D'
)
