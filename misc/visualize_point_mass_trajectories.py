import os
import sys
import argparse
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from stable_baselines.trpo_mpi import TRPO

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

from deep_sprl.experiments import PointMassExperiment, CurriculumType

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}')
FONT_SIZE = 7

labels = {"default": "Default", "random": "Random", "alp_gmm": "ALP-GMM", "self_paced": "SPDL", "goal_gan": "GoalGAN"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_log_dir", type=str, default="logs")
    parser.add_argument("--learner", type=str, required=True, choices=["trpo", "ppo", "sac"])
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    # Create the evaluation environment
    exp = PointMassExperiment(args.base_log_dir, "default", args.learner, {}, 1)
    log_dir = os.path.join(os.path.dirname(__file__), "..", args.base_log_dir, "point_mass")
    types = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    for cur_type in types:
        f = plt.figure(figsize=(1.4, 1.4))
        ax = f.gca()
        ax.plot([-5., 2.], [-0.1, -0.1], linewidth=5, color="black")
        ax.plot([3., 5.], [-0.1, -0.1], linewidth=5, color="black")
        ax.plot([-0.25, 0.25], [-3.25, -2.75], linewidth=3, color="red")
        ax.plot([-0.25, 0.25], [-2.75, -3.25], linewidth=3, color="red")

        exp.curriculum = CurriculumType.from_string(cur_type)
        type_log_dir = os.path.join(os.path.dirname(__file__), "..", os.path.dirname(exp.get_log_dir()))
        seeds = [int(d.split("-")[1]) for d in os.listdir(type_log_dir) if os.path.isdir(os.path.join(type_log_dir, d))]
        for seed in seeds:
            path = os.path.join(type_log_dir, "seed-" + str(seed), "iteration-" + str(995))
            if os.path.exists(path):
                model = exp.learner.load_for_evaluation(os.path.join(path, "model"), exp.vec_eval_env)

                path = []
                done = False
                obs = exp.vec_eval_env.reset()
                path.append(obs[0][[0, 2]])
                while not done:
                    action = model.step(obs, state=None, deterministic=False)
                    obs, reward, done, info = exp.vec_eval_env.step(action)

                    # We need to add this check because the vectorized environment automatically resets everything on
                    # done
                    if not done:
                        path.append(obs[0][[0, 2]])

                path = np.array(path)
                ax.plot(path[:, 0], path[:, 1], color="C0", alpha=0.5, linewidth=3)

        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])
        ax.set_xticks([])
        ax.set_yticks([])

        if args.save_dir is None:
            plt.title(labels[cur_type])
            plt.tight_layout()
            plt.show()
        else:
            plt.tight_layout()
            plt.savefig(os.path.join(args.save_dir, "point_mass_%s_%s_trajs.pdf" % (args.learner, cur_type)),
                        bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    main()
