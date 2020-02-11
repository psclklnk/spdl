import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import os
from matplotlib import rc

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

FONT_SIZE = 7
KL_DICT = {"point_mass": [8 * 10 ** -4, 2 * 10 ** 6], "point_mass_2d": [8 * 10 ** -4, 2 * 10 ** 6],
           "ball_catching": [5 * 10 ** -3, 1 * 10 ** 1]}
METHODS = ["self_paced", "alp_gmm", "random", "default", "goal_gan"]
LABEL_DICT = {"self_paced": r"SPDL", "alp_gmm": r"ALP-GMM", "random": r"Random", "default": r"Default",
              "goal_gan": r"GoalGAN"}
COLOR_DICT = {"self_paced": "C0", "default": "C1", "random": "C2", "alp_gmm": "C3", "goal_gan": "C4"}
LIMITS = {"point_mass": [-0.1, 10.1], "point_mass_2d": [-0.1, 10.1]}


def add_plot(base_log_dir, ax, color, standard_error=False, max_seed=None):
    self_paced = "self-paced" in base_log_dir or "self_paced" in base_log_dir
    iterations = []
    seed_performances = []
    kl_divergences = []

    if max_seed is None:
        max_seed = int(1e6)

    seeds = [int(d.split("-")[1]) for d in os.listdir(base_log_dir) if d.startswith("seed-")]
    for seed in [s for s in seeds if s <= max_seed]:
        seed_dir = "seed-" + str(seed)
        seed_log_dir = os.path.join(base_log_dir, seed_dir)

        iteration_dirs = [d for d in os.listdir(seed_log_dir) if d.startswith("iteration-")]
        unsorted_iterations = np.array([int(d[len("iteration-"):]) for d in iteration_dirs])
        idxs = np.argsort(unsorted_iterations)
        iterations = unsorted_iterations[idxs]

        if os.path.exists(os.path.join(seed_log_dir, "performance.pkl")):
            with open(os.path.join(seed_log_dir, "performance.pkl"), "rb") as f:
                seed_performances.append(pickle.load(f))
        else:
            pass
            # raise RuntimeError("No Performance log was found")

        if self_paced:
            if os.path.exists(os.path.join(seed_log_dir, "kl_divergences.pkl")):
                with open(os.path.join(seed_log_dir, "kl_divergences.pkl"), "rb") as f:
                    kl_divergences.append(pickle.load(f))
            else:
                pass
                # raise RuntimeError("No KL-Divergence log was found")

    if len(seed_performances) > 0:
        min_length = np.min([len(seed_performance) for seed_performance in seed_performances])
        iterations = iterations[0: min_length]
        seed_performances = [seed_performance[0: min_length] for seed_performance in seed_performances]

        if standard_error:
            mid = np.mean(seed_performances, axis=0)
            sem = np.std(seed_performances, axis=0) / np.sqrt(len(seed_performances))
            low = mid - 2 * sem
            high = mid + 2 * sem
        else:
            low = np.percentile(seed_performances, 5, axis=0)
            mid = np.percentile(seed_performances, 50, axis=0)
            high = np.percentile(seed_performances, 95, axis=0)

        l, = ax.plot(iterations, mid, color=color, linewidth=2.0)
        ax.fill_between(iterations, low, high, color=color, alpha=0.5)
        return l
    else:
        return None


def add_kl_plot(base_log_dir, ax, kl_limits=None, standard_error=False, yticks=None):
    kl_divergences = []
    for seed_dir in [d for d in os.listdir(base_log_dir) if d.startswith("seed-")]:
        seed_log_dir = os.path.join(base_log_dir, seed_dir)

        iteration_dirs = [d for d in os.listdir(seed_log_dir) if d.startswith("iteration-")]
        unsorted_iterations = np.array([int(d[len("iteration-"):]) for d in iteration_dirs])
        idxs = np.argsort(unsorted_iterations)
        iterations = unsorted_iterations[idxs]

        if os.path.exists(os.path.join(seed_log_dir, "kl_divergences.pkl")):
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
    if standard_error:
        kl_mid = np.mean(kl_divergences, axis=0)
        sem = np.std(kl_divergences, axis=0) / np.sqrt(len(kl_divergences))
        kl_low = kl_mid - sem
        kl_high = kl_mid + sem
    else:
        kl_low = np.percentile(kl_divergences, 5, axis=0)
        kl_mid = np.percentile(kl_divergences, 50, axis=0)
        kl_high = np.percentile(kl_divergences, 95, axis=0)

    ax.plot(iterations, kl_mid, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
    ax.fill_between(iterations, kl_low, kl_high, color="black", alpha=0.1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_log_dir", type=str, default="logs")
    parser.add_argument("--env", type=str, default="point_mass",
                        choices=["point_mass", "point_mass_2d", "ball_catching"])

    args = parser.parse_args()
    exp_dir = os.path.join(args.base_log_dir, args.env)

    if not os.path.exists(exp_dir):
        print("No experiment directory was found under: " + exp_dir)
        return

    f, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [2.5, 1.5]}, figsize=(3, 2.5))

    axs[0].set_ylabel(r"Performance", fontsize=FONT_SIZE)
    axs[1].set_ylabel(r"KL-Divergence", fontsize=FONT_SIZE)
    axs[1].set_xlabel(r"Iteration", fontsize=FONT_SIZE)

    for i in range(0, 2):
        axs[i].tick_params(axis='both', which='major', labelsize=FONT_SIZE)
        axs[i].tick_params(axis='both', which='minor', labelsize=FONT_SIZE)

    lines = []
    labels = []
    for method in METHODS:
        method_dir = os.path.join(exp_dir, method)
        if os.path.exists(method_dir):
            line = add_plot(method_dir, axs[0], COLOR_DICT[method], standard_error=True)
            if line is not None:
                lines.append(line)
                labels.append(LABEL_DICT[method])
            if method == "self_paced":
                add_kl_plot(method_dir, axs[1], KL_DICT[args.env])
    axs[0].legend(lines, labels, fontsize=FONT_SIZE, framealpha=0.3)
    if args.env in LIMITS:
        axs[0].set_ylim(LIMITS[args.env])
    axs[0].grid()
    axs[1].grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
