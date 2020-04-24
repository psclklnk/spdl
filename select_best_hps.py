import os
import pickle
import argparse
import numpy as np


def get_name(hp_run, learner):
    if hp_run == learner:
        return "default"
    else:
        return hp_run[len(learner) + 1:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_log_dir", type=str, default="logs")
    parser.add_argument("--env", type=str, default="point_mass",
                        choices=["point_mass", "point_mass_2d", "ball_catching"])
    parser.add_argument("--learner", type=str, default="trpo", choices=["trpo", "ppo", "sac", "all"])
    parser.add_argument("--type", type=str, default="alp_gmm")

    args, remainder = parser.parse_known_args()
    exp_dir = os.path.join(args.base_log_dir, args.env, args.type)

    learners = []
    if args.learner == "all":
        learners.extend(["trpo", "ppo", "sac"])
    else:
        learners.append(args.learner)

    # In the first run, build the dictionary for lookup
    all_data = []
    for learner in learners:
        data = {}
        hp_runs = [d for d in os.listdir(exp_dir) if d.startswith(learner)]
        for hp_run in hp_runs:
            seeds = [sd for sd in os.listdir(os.path.join(exp_dir, hp_run)) if sd.startswith("seed-")]
            auc = 0.
            for seed in seeds:
                seed_dir = os.path.join(exp_dir, hp_run, seed)
                with open(os.path.join(seed_dir, "performance.pkl"), "rb") as f:
                    auc += np.sum(pickle.load(f))
            auc /= len(seeds)
            data[get_name(hp_run, learner)] = auc
        all_data.append(data)

        print("Result for " + learner)
        for k in sorted(data, key=data.get)[-5:]:
            print("%s: %.3E" % (k, data[k]))
        print("")

    combined_data = {}
    for data in all_data:
        for k, v in data.items():
            if k not in combined_data:
                combined_data[k] = [v]
            else:
                combined_data[k].append(v)

    to_remove = []
    for k, v in combined_data.items():
        if len(combined_data[k]) != len(learners):
            print("Removing hp '" + k + "' as it was not present for all learning algorithms")
            to_remove.append(k)

    for k in to_remove:
        del combined_data[k]

    for k in combined_data.keys():
        combined_data[k] = np.mean(combined_data[k])

    print("Average best performance:")
    for k in sorted(combined_data, key=combined_data.get)[-5:]:
        # Get the performance in the individual runs
        single_performances = []
        for data in all_data:
            count = 1
            for k_sub in reversed(sorted(data, key=data.get)):
                if k_sub == k:
                    single_performances.append(count)
                    break
                else:
                    count += 1

        print("%s:\t\t%.3E\t\t\t%s" % (k, combined_data[k], str(single_performances)))


if __name__ == "__main__":
    main()
