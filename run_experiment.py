import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import argparse
import deep_sprl.environments
from deep_sprl.util.parameter_parser import parse_parameters


def main():
    parser = argparse.ArgumentParser("Self-Paced Learning experiment runner")
    parser.add_argument("--base_log_dir", type=str, default="logs")
    parser.add_argument("--type", type=str, default="default",
                        choices=["default", "random", "self_paced", "alp_gmm", "goal_gan"])
    parser.add_argument("--learner", type=str, choices=["trpo", "ppo", "sac"])
    parser.add_argument("--env", type=str, default="point_mass",
                        choices=["point_mass", "point_mass_2d", "ball_catching"])
    parser.add_argument("--seed", type=int, default=1)

    args, remainder = parser.parse_known_args()
    parameters = parse_parameters(remainder)

    if args.type == "self_paced":
        import torch
        torch.set_num_threads(1)

    if args.env == "point_mass":
        from deep_sprl.experiments import PointMassExperiment
        exp = PointMassExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed)
    elif args.env == "point_mass_2d":
        from deep_sprl.experiments import PointMass2DExperiment
        exp = PointMass2DExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed)
    else:
        from deep_sprl.experiments import BallCatchingExperiment
        exp = BallCatchingExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed)

    exp.train()
    exp.evaluate()


if __name__ == "__main__":
    main()
