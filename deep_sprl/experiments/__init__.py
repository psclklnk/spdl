# Do this import to ensure that the Gym environments get registered properly
import deep_sprl.environments
from .point_mass_experiment import PointMassExperiment
from .point_mass_2d_experiment import PointMass2DExperiment
from .abstract_experiment import CurriculumType, Learner

__all__ = ['CurriculumType', 'PointMassExperiment', 'PointMass2DExperiment', 'Learner']
try:
    import mujoco_py
    from .ball_catching_experiment import BallCatchingExperiment
except ModuleNotFoundError:
    from .ball_catching_experiment import DummyBallCatchingExperiment as BallCatchingExperiment

    pass

__all__ += ['BallCatchingExperiment']