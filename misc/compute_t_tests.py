import os
import sys
import pickle
import argparse
import numpy as np
from scipy.stats import ttest_ind

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

from deep_sprl.util.parameter_parser import parse_parameters
from deep_sprl.experiments import CurriculumType


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_log_dir", type=str, default="logs")
    parser.add_argument("--env", type=str, default="point_mass",
                        choices=["point_mass", "point_mass_2d", "ball_catching", "ant"])
    parser.add_argument("--learner", type=str, default="trpo", choices=["trpo", "ppo", "sac"])

    args, remainder = parser.parse_known_args()
    parameters = parse_parameters(remainder)
    if args.env == "point_mass":
        from deep_sprl.experiments import PointMassExperiment
        exp = PointMassExperiment(args.base_log_dir, "default", args.learner, parameters, 1)
    elif args.env == "point_mass_2d":
        from deep_sprl.experiments import PointMass2DExperiment
        exp = PointMass2DExperiment(args.base_log_dir, "default", args.learner, parameters, 1)
    elif args.env == "ant":
        from deep_sprl.experiments import AntExperiment
        exp = AntExperiment(args.base_log_dir, "default", args.learner, parameters, 1)
    else:
        from deep_sprl.experiments import BallCatchingExperiment
        exp = BallCatchingExperiment(args.base_log_dir, "default", args.learner, parameters, 1)

    log_dir = os.path.join(os.path.dirname(__file__), "..", args.base_log_dir, args.env)
    types = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    if args.env == "ball_catching":
        exps = [exp, BallCatchingExperiment(args.base_log_dir, "default", args.learner, {"INIT_CONTEXT": False}, 1),
                BallCatchingExperiment(args.base_log_dir, "default", args.learner, {"INIT_POLICY": False}, 1)]
        appendices = ["", "*", "*"]
    else:
        exps = [exp]
        appendices = [""]

    performances = {}
    for cur_type in types:
        for exp, appendix in zip(exps, appendices):
            exp.curriculum = CurriculumType.from_string(cur_type)
            type_log_dir = os.path.join(os.path.dirname(__file__), "..", os.path.dirname(exp.get_log_dir()))
            if os.path.exists(type_log_dir):
                seeds = [int(d.split("-")[1]) for d in os.listdir(type_log_dir) if
                         os.path.isdir(os.path.join(type_log_dir, d))]
                if len(seeds) != 0:
                    type_perf = []
                    for seed in seeds:
                        seed_log_dir = os.path.join(type_log_dir, "seed-" + str(seed))
                        with open(os.path.join(seed_log_dir, "performance.pkl"), "rb") as f:
                            type_perf.append(pickle.load(f)[-1])
                    performances[cur_type + appendix] = np.array(type_perf)

    best_type = None
    best_mean_perf = -np.inf
    best_se = None
    for key, value in performances.items():
        if np.mean(value) > best_mean_perf:
            best_mean_perf = np.mean(value)
            best_se = np.std(value) / np.sqrt(len(value))
            best_type = key

    print("Best Type: %s, Best performance: %.2f, std: %.2f" % (best_type, best_mean_perf, best_se))
    for key in sorted(performances.keys()):
        if key != best_type:
            mean_perf = np.mean(performances[key])
            se = np.std(performances[key], axis=0) / np.sqrt(len(performances[key]))
            pvalue = ttest_ind(performances[best_type], performances[key])[1]
            print("Type: %s, performance: %.2f, std: %.2f, P-Value: %.3e" % (key, mean_perf, se, pvalue))


if __name__ == "__main__":
    main()
