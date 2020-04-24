import os
import pickle
import argparse
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from deep_sprl.util.parameter_parser import parse_parameters
from deep_sprl.experiments import CurriculumType

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

FONT_SIZE = 7
KL_DICT = {"point_mass": [8 * 10 ** -4, 2 * 10 ** 6], "point_mass_2d": [8 * 10 ** -4, 2 * 10 ** 6],
           "ball_catching": [5 * 10 ** -3, 1 * 10 ** 1], "ant": [8 * 10 ** -4, 2 * 10 ** 6]}
METHODS = ["self_paced", "alp_gmm", "random", "default", "goal_gan"]
LABEL_DICT = {"self_paced": r"SPDL", "alp_gmm": r"ALP-GMM", "random": r"Random", "default": r"Default",
              "goal_gan": r"GoalGAN", "sprl": r"SPRL"}
COLOR_DICT = {"self_paced": "C0", "default": "C1", "random": "C2", "alp_gmm": "C3", "goal_gan": "C4", "sprl": "C8"}
LIMITS = {"point_mass": [-0.1, 10.1], "point_mass_2d": [-0.1, 10.1]}


def add_sprl_plot(exp, axs, lines, kl_lines, labels, kl_labels, base_log_dir, color, max_seed=None):
    iteration_dirs = [d for d in os.listdir(exp.get_log_dir()) if d.startswith("iteration-")]
    unsorted_iterations = np.array([int(d[len("iteration-"):]) for d in iteration_dirs])
    iterations = unsorted_iterations[np.argsort(unsorted_iterations)]
    seed_performances = []
    kl_divergences = []

    if max_seed is None:
        max_seed = int(1e6)

    base_dir = os.path.join(base_log_dir, exp.get_env_name(), "sprl")
    seeds = [int(d.split("-")[1]) for d in os.listdir(base_dir) if d.startswith("seed-")]
    for seed in [s for s in seeds if s <= max_seed]:
        seed_dir = "seed-" + str(seed)
        seed_log_dir = os.path.join(base_dir, seed_dir)

        if os.path.exists(os.path.join(seed_log_dir, "performances.pkl")):
            with open(os.path.join(seed_log_dir, "performances.pkl"), "rb") as f:
                counts, performances = pickle.load(f)

            seed_iters = []
            seed_perfs = []
            for perf, count in zip(performances, counts):
                seed_iters.append(float(count) / exp.STEPS_PER_ITER)
                seed_perfs.append(perf)

            seed_performances.append(np.interp(iterations, seed_iters, seed_perfs))
        else:
            pass
            # raise RuntimeError("No Performance log was found")

        if os.path.exists(os.path.join(seed_log_dir, "kl-divergences.pkl")):
            with open(os.path.join(seed_log_dir, "kl-divergences.pkl"), "rb") as f:
                counts, kls = pickle.load(f)

            seed_iters = []
            seed_kls = []
            for kl, count in zip(kls, counts):
                seed_iters.append(float(count) / exp.STEPS_PER_ITER)
                seed_kls.append(kl)

            kl_divergences.append(np.interp(iterations, seed_iters, seed_kls))
        else:
            pass
            # raise RuntimeError("No KL-Divergence log was found")

    if len(seed_performances) > 0:
        mid = np.mean(seed_performances, axis=0)
        sem = np.std(seed_performances, axis=0) / np.sqrt(len(seed_performances))
        low = mid - 2 * sem
        high = mid + 2 * sem

        l, = axs[0].plot(iterations, mid, color=color, linewidth=2.0)
        axs[0].fill_between(iterations, low, high, color=color, alpha=0.5)
        lines.append(l)
        labels.append(LABEL_DICT["sprl"])

    if len(kl_divergences) > 0:
        kl_low = np.percentile(kl_divergences, 5, axis=0)
        kl_mid = np.percentile(kl_divergences, 50, axis=0)
        kl_high = np.percentile(kl_divergences, 95, axis=0)

        l, = axs[1].plot(iterations, kl_mid, color=color, linewidth=2.0)
        axs[1].fill_between(iterations, kl_low, kl_high, color=color, alpha=0.5)
        kl_lines.append(l)
        kl_labels.append(LABEL_DICT["sprl"])

    return lines if len(lines) > 0 else None


def add_plot(base_log_dir, ax, color, max_seed=None):
    iterations = []
    seed_performances = []

    if max_seed is None:
        max_seed = int(1e6)

    seeds = [int(d.split("-")[1]) for d in os.listdir(base_log_dir) if d.startswith("seed-")]
    for seed in [s for s in seeds if s <= max_seed]:
        seed_dir = "seed-" + str(seed)
        seed_log_dir = os.path.join(base_log_dir, seed_dir)

        if os.path.exists(os.path.join(seed_log_dir, "performance.pkl")):
            iteration_dirs = [d for d in os.listdir(seed_log_dir) if d.startswith("iteration-")]
            unsorted_iterations = np.array([int(d[len("iteration-"):]) for d in iteration_dirs])
            idxs = np.argsort(unsorted_iterations)
            iterations = unsorted_iterations[idxs]

            with open(os.path.join(seed_log_dir, "performance.pkl"), "rb") as f:
                seed_performances.append(pickle.load(f))
        else:
            pass
            # raise RuntimeError("No Performance log was found")

    if len(seed_performances) > 0:
        min_length = np.min([len(seed_performance) for seed_performance in seed_performances])
        iterations = iterations[0: min_length]
        seed_performances = [seed_performance[0: min_length] for seed_performance in seed_performances]

        mid = np.mean(seed_performances, axis=0)
        sem = np.std(seed_performances, axis=0) / np.sqrt(len(seed_performances))
        low = mid - 2 * sem
        high = mid + 2 * sem

        l, = ax.plot(iterations, mid, color=color, linewidth=2.0)
        ax.fill_between(iterations, low, high, color=color, alpha=0.5)
        return l
    else:
        return None


def add_kl_plot(base_log_dir, ax, kl_limits=None, yticks=None, kl_color="black"):
    kl_divergences = []
    for seed_dir in [d for d in os.listdir(base_log_dir) if d.startswith("seed-")]:
        seed_log_dir = os.path.join(base_log_dir, seed_dir)

        if os.path.exists(os.path.join(seed_log_dir, "kl_divergences.pkl")):
            iteration_dirs = [d for d in os.listdir(seed_log_dir) if d.startswith("iteration-")]
            unsorted_iterations = np.array([int(d[len("iteration-"):]) for d in iteration_dirs])
            idxs = np.argsort(unsorted_iterations)
            iterations = unsorted_iterations[idxs]
            with open(os.path.join(seed_log_dir, "kl_divergences.pkl"), "rb") as f:
                kl_divergences.append(pickle.load(f))
        else:
            pass
            # raise RuntimeError("No KL-Divergence log was found")

    min_length = np.min([len(kl_divergence) for kl_divergence in kl_divergences])

    ax.set_yscale("log")
    ax.set_ylim(kl_limits)

    if yticks is not None:
        ax.set_yticks(yticks)

    kl_divergences = [kl_divergence[0: min_length] for kl_divergence in kl_divergences]
    kl_low = np.percentile(kl_divergences, 5, axis=0)
    kl_mid = np.percentile(kl_divergences, 50, axis=0)
    kl_high = np.percentile(kl_divergences, 95, axis=0)

    line, = ax.plot(iterations, kl_mid, color=kl_color, linewidth=2.0)
    ax.fill_between(iterations, kl_low, kl_high, color=kl_color, alpha=0.5)

    return line


def add_plots(exp, axs, lines, kl_lines, labels, kl_labels, env_name, kl_color="black"):
    for method in METHODS:
        exp.curriculum = CurriculumType.from_string(method)
        method_dir = os.path.dirname(exp.get_log_dir())
        if os.path.exists(method_dir):
            line = add_plot(method_dir, axs[0], COLOR_DICT[method])
            if line is not None:
                lines.append(line)
                labels.append(LABEL_DICT[method])
            if method == "self_paced":
                kl_line = add_kl_plot(method_dir, axs[1], KL_DICT[env_name], kl_color=kl_color)
                if kl_line is not None:
                    kl_lines.append(kl_line)
                    kl_labels.append(LABEL_DICT[method])


def main():
    global LABEL_DICT
    global COLOR_DICT

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

    # (3, 2.8)
    f, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [2.8, 1.2]}, figsize=(3, 2.5))

    axs[0].set_ylabel(r"Performance", fontsize=FONT_SIZE)
    axs[1].set_ylabel(r"KL-Divergence", fontsize=FONT_SIZE)
    axs[1].set_xlabel(r"Iteration", fontsize=FONT_SIZE)

    for i in range(0, 2):
        axs[i].tick_params(axis='both', which='major', labelsize=FONT_SIZE)
        axs[i].tick_params(axis='both', which='minor', labelsize=FONT_SIZE)

    lines = []
    kl_lines = []
    labels = []
    kl_labels = []
    if args.env != "ball_catching":
        add_plots(exp, axs, lines, kl_lines, labels, kl_labels, args.env, kl_color="C0")
        if args.env.startswith("point_mass"):
            add_sprl_plot(exp, axs, lines, kl_lines, labels, kl_labels, args.base_log_dir, COLOR_DICT["sprl"])
    else:
        LABEL_DICT["self_paced"] = r"SPDL*"
        LABEL_DICT["goal_gan"] = r"GoalGAN*"
        COLOR_DICT["self_paced"] = "C5"
        COLOR_DICT["goal_gan"] = "C8"
        add_plots(exp, axs, lines, kl_lines, labels, kl_labels, args.env, kl_color="C5")
        exp = BallCatchingExperiment(args.base_log_dir, "default", args.learner, {"INIT_CONTEXT": False}, 1)
        LABEL_DICT = {"self_paced": r"SPDL", "goal_gan": r"GoalGAN"}
        COLOR_DICT = {"self_paced": "C0", "goal_gan": "C4"}
        add_plots(exp, axs, lines, kl_lines, labels, kl_labels, args.env, kl_color="C0")

        exp = BallCatchingExperiment(args.base_log_dir, "default", args.learner, {"INIT_POLICY": False}, 1)
        LABEL_DICT = {"default": r"Default*"}
        COLOR_DICT = {"default": "C7"}
        add_plots(exp, axs, lines, kl_lines, labels, kl_labels, args.env)
    axs[0].legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize=FONT_SIZE,
                  handlelength=1.0, labelspacing=0., handletextpad=0.5, columnspacing=1.0)
    # axs[1].legend(kl_lines, kl_labels, fontsize=FONT_SIZE, framealpha=0.3)
    if args.env in LIMITS:
        axs[0].set_ylim(LIMITS[args.env])
    axs[0].grid()
    axs[1].grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
