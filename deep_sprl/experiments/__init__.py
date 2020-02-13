from .point_mass_experiment import PointMassExperiment
from .point_mass_2d_experiment import PointMass2DExperiment
try:
    import mujoco_py
    from .ball_catching_experiment import BallCatchingExperiment
    __all__ = ['BallCatchingExperiment', 'PointMassExperiment', 'PointMass2DExperiment']
except ModuleNotFoundError:
    __all__ = ['PointMassExperiment', 'PointMass2DExperiment']


