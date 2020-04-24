import os
import sys
import pickle
import argparse
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

from deep_sprl.experiments import BallCatchingExperiment, CurriculumType

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
FONT_SIZE = 7

labels = {"default": "Default", "goal_gan": "Goal GAN", "alp_gmm": "ALP-GMM", "self_paced": "SPDL",
          "self_paced*": "SPDL*", "goal_gan*": "GoalGAN*", "default*": "Default*"}
colors = {"self_paced": "C0", "default": "C1", "random": "C2", "alp_gmm": "C3", "goal_gan": "C4", "self_paced*": "C5",
          "goal_gan*": "C8", "default*": "C7"}
priorities = {"self_paced": 5, "default": 0, "alp_gmm": 4, "goal_gan": 2, "self_paced*": 6,
              "goal_gan*": 3, "default*": 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_log_dir", type=str, default="logs")
    parser.add_argument("--learner", type=str, required=True, choices=["trpo", "ppo", "sac"])
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    # We just use this to create the evaluation environment
    exps = [BallCatchingExperiment(args.base_log_dir, "default", args.learner, {}, 1),
            BallCatchingExperiment(args.base_log_dir, "default", args.learner, {"INIT_CONTEXT": False}, 1),
            BallCatchingExperiment(args.base_log_dir, "default", args.learner, {"INIT_POLICY": False}, 1)]
    suffixes = ["", "*", "*"]
    log_dir = os.path.join(os.path.dirname(__file__), "..", args.base_log_dir, "ball_catching")
    types = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    for exp, suffix in zip(exps, suffixes):
        for cur_type in types:
            exp.curriculum = CurriculumType.from_string(cur_type)
            type_log_dir = os.path.join(os.path.dirname(__file__), "..", os.path.dirname(exp.get_log_dir()))
            if os.path.exists(type_log_dir):
                seeds = [int(d.split("-")[1]) for d in os.listdir(type_log_dir) if
                         os.path.isdir(os.path.join(type_log_dir, d))]
                for seed in seeds:
                    print("Starting evaluation for method '" + str(cur_type) + "' and seed " + str(seed))
                    seed_log_dir = os.path.join(type_log_dir, "seed-" + str(seed))
                    seed_successes = []
                    if not os.path.exists(os.path.join(seed_log_dir, "catch_stats.pkl")):
                        model_load_path = os.path.join(seed_log_dir, "iteration-495")
                        if os.path.exists(model_load_path):
                            model = exp.learner.load_for_evaluation(
                                os.path.join(seed_log_dir, "iteration-495", "model"),
                                exp.eval_env)

                            successes = []
                            for i in range(0, 200):
                                obs = exp.eval_env.reset()
                                done = False
                                while not done:
                                    action = model.step(obs, state=None, deterministic=False)
                                    obs, rewards, done, infos = exp.eval_env.step(action)
                                successes.append(infos['success'])

                            seed_successes.append(np.mean(np.array(successes).astype(np.float)))
                            print("Evaluated: " + str(seed_successes[-1]))
                            with open(os.path.join(seed_log_dir, "catch_stats.pkl"), "wb") as f:
                                pickle.dump(np.array(seed_successes), f)

    data_dict = {}
    for exp, suffix in zip(exps, suffixes):
        for cur_type in types:
            exp.curriculum = CurriculumType.from_string(cur_type)
            type_log_dir = os.path.join(os.path.dirname(__file__), "..", os.path.dirname(exp.get_log_dir()))
            if os.path.exists(type_log_dir):
                seeds = [int(d.split("-")[1]) for d in os.listdir(type_log_dir) if
                         os.path.isdir(os.path.join(type_log_dir, d))]
                cur_successes = []
                for seed in seeds:
                    seed_log_dir = os.path.join(type_log_dir, "seed-" + str(seed))
                    with open(os.path.join(seed_log_dir, "catch_stats.pkl"), "rb") as f:
                        cur_successes.append(pickle.load(f))
                data_dict[cur_type + suffix] = (np.mean(np.squeeze(cur_successes)),
                                                np.std(np.squeeze(cur_successes)) / np.sqrt(len(cur_successes)))

    mean_successes = []
    ste_successes = []
    ls = []
    cs = []
    for key, value in sorted(data_dict.items(), key=lambda x: priorities[x[0]]):
        mean_successes.append(value[0])
        ste_successes.append(value[1])
        ls.append(labels[key])
        cs.append(colors[key])

    f = plt.figure(figsize=(5, 1.75))
    ax = f.gca()
    for i in range(0, 2):
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
        ax.tick_params(axis='both', which='minor', labelsize=FONT_SIZE)

    plt.grid(zorder=0)
    plt.bar(np.arange(len(ls)), mean_successes, yerr=ste_successes, capsize=10, zorder=3, color=cs)
    plt.xticks(np.arange(len(ls)), ls)
    plt.yticks([0., 0.25, 0.5, 0.75])
    plt.ylabel("Catching Rate", fontsize=FONT_SIZE)
    plt.ylim([plt.ylim()[0], 0.9])

    plt.tight_layout()
    if args.save_dir is None:
        plt.show()
    else:
        plt.savefig(os.path.join(args.save_dir, "ball_catching_stats_%s.pdf" % args.learner), bbox_inches='tight',
                    pad_inches=0)


if __name__ == "__main__":
    main()
