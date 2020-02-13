import argparse
import deep_sprl.environments


def main():
    parser = argparse.ArgumentParser("Self-Paced Learning experiment runner")
    parser.add_argument("--base_log_dir", type=str, default="logs")
    parser.add_argument("--type", type=str, default="default",
                        choices=["default", "random", "self_paced", "alp_gmm", "goal_gan"])
    parser.add_argument("--env", type=str, default="point_mass",
                        choices=["point_mass", "point_mass_2d", "ball_catching"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--view", action="store_true")

    args = parser.parse_args()

    if args.env == "point_mass":
        from deep_sprl.experiments import PointMassExperiment
        exp = PointMassExperiment(args.base_log_dir, args.type, {}, args.seed)
    elif args.env == "point_mass_2d":
        from deep_sprl.experiments import PointMass2DExperiment
        exp = PointMass2DExperiment(args.base_log_dir, args.type, {}, args.seed)
    else:
        from deep_sprl.experiments import BallCatchingExperiment
        exp = BallCatchingExperiment(args.base_log_dir, args.type, {}, args.seed)

    exp.train()
    exp.evaluate()


if __name__ == "__main__":
    main()
