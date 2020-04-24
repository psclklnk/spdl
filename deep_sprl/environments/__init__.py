from gym.envs.registration import register
try:
    import mujoco_py
    register(
        id='ContextualBallCatching-v1',
        max_episode_steps=200,
        entry_point='deep_sprl.environments.contextual_ball_catching:ContextualBallCatching'
    )
except ModuleNotFoundError:
    pass

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
