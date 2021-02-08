import os
import pickle
import argparse
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.transforms import Bbox
from deep_sprl.util.parameter_parser import parse_parameters
from deep_sprl.experiments import CurriculumType
from deep_sprl.util.gaussian_torch_distribution import GaussianTorchDistribution

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)

FONT_SIZE = 7
KL_DICT = {"point_mass": [8 * 10 ** -4, 2 * 10 ** 6], "point_mass_2d": [8 * 10 ** -4, 2 * 10 ** 6],
           "ball_catching": [5 * 10 ** -3, 1 * 10 ** 1]}
METHODS = ["self_paced", "self_paced_v2", "alp_gmm", "random", "default", "goal_gan"]
LABEL_DICT = {"self_paced": r"SPDL", "self_paced_v2": r"SPDL2", "alp_gmm": r"ALP-GMM", "random": r"Random",
              "default": r"Default", "goal_gan": r"GoalGAN", "sprl": r"SPRL"}
COLOR_DICT = {"self_paced": "C0", "self_paced_v2": "C1", "default": "C1", "random": "C2", "alp_gmm": "C3",
              "goal_gan": "C4", "sprl": "C8"}
MARKER_DICT = {"self_paced": "^", "default": "o", "random": "s", "alp_gmm": "<", "goal_gan": "D", "self_paced_v2": "+"}
MARKEVERY = {"point_mass": 6, "point_mass_2d": 6, "ball_catching": 3}
LIMITS = {"point_mass": [-0.1, 10.1], "point_mass_2d": [-0.1, 10.1]}
ENV_NAMES = {"ball_catching": "Ball Catching", "point_mass": "Point Mass",
             "point_mass_2d": "Point Mass (2D)"}

N_DIST_ITERS = 6
DIST_ITERS = {"point_mass_2d": [0, 20, 30, 50, 65, 120], "ball_catching": [0, 50, 80, 110, 150, 200]}
DIST_SEEDS = {"point_mass_2d": [5, 11, 16], "ball_catching": [2, 7, 16]}
DIST_PROJECTIONS = {"point_mass_2d": lambda x: x,
                    "ball_catching": lambda x: np.array([-np.cos(x[0]) * x[1], 0.75 + np.sin(x[0]) * x[1]])}
DIST_XTICKS = {"point_mass_2d": [-4, -2, 0, 2, 4], "ball_catching": [-1, -0.5, 0]}
DIST_YTICKS = {"point_mass_2d": [0.5, 3, 5.5, 8], "ball_catching": [1, 1.4, 1.8]}
DIST_XLABELS = {"point_mass_2d": "Position", "ball_catching": "X-Position"}
DIST_YLABELS = {"point_mass_2d": "Width", "ball_catching": "Y-Position"}
DIST_NSAMPLES = {"point_mass_2d": 20, "ball_catching": 40}
DIST_SHOW_MEAN = {"point_mass_2d": True, "ball_catching": False}

WIDTH = 5.6
MUL = 0.4
BBOXES = {"point_mass+point_mass_2d": Bbox(np.array([[0.22, 0.], [WIDTH - 0.22, MUL * WIDTH + 0.15]]))}


def add_sprl_plot(exp, ax, lines, labels, base_log_dir, color, max_seed=None):
    if max_seed is None:
        max_seed = int(1e6)

    base_dir = os.path.join(base_log_dir, exp.get_env_name(), "sprl")
    if os.path.exists(base_dir):
        iteration_dirs = [d for d in os.listdir(exp.get_log_dir()) if d.startswith("iteration-")]
        unsorted_iterations = np.array([int(d[len("iteration-"):]) for d in iteration_dirs])
        iterations = unsorted_iterations[np.argsort(unsorted_iterations)]
        seed_performances = []
        kl_divergences = []

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

            l, = ax.plot(iterations, mid, color=color, linewidth=2.0)
            ax.fill_between(iterations, low, high, color=color, alpha=0.5)
            lines.append(l)
            labels.append(LABEL_DICT["sprl"])

        return lines if len(lines) > 0 else None
    else:
        return None


def add_plot(base_log_dir, ax, color, max_seed=None, marker="o", markevery=3):
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
        print("Found %d completed seeds" % len(seed_performances))
        min_length = np.min([len(seed_performance) for seed_performance in seed_performances])
        iterations = iterations[0: min_length]
        seed_performances = [seed_performance[0: min_length] for seed_performance in seed_performances]

        mid = np.mean(seed_performances, axis=0)
        sem = np.std(seed_performances, axis=0) / np.sqrt(len(seed_performances))
        low = mid - 2 * sem
        high = mid + 2 * sem

        l, = ax.plot(iterations, mid, color=color, linewidth=1, marker=marker, markersize=2, markevery=markevery)
        ax.fill_between(iterations, low, high, color=color, alpha=0.5)
        return l
    else:
        return None


def visualize_distribution(exp, ax, iteration, alpha, cval, xticks, yticks, xlabel=None, ylabel=None,
                           seeds=None, n_samples=5, hide_y_ticks=False, projection=None, markers=None,
                           scatter_mean=False):
    jet = plt.get_cmap("hot")
    c_norm = colors.Normalize(vmin=0, vmax=1)
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=jet)

    dim = exp.LOWER_CONTEXT_BOUNDS.shape[0]
    if seeds is None:
        seeds = range(1, 21)

    if isinstance(cval, float):
        cval = [cval] * len(seeds)

    if markers is None:
        markers = ["o"] * len(seeds)

    for seed, cv, marker in zip(seeds, cval, markers):
        dist = os.path.join(os.path.dirname(exp.get_log_dir()), "seed-%d" % seed, "iteration-%d" % iteration,
                            "context_dist.npy")
        dist = GaussianTorchDistribution.from_weights(dim, np.load(dist))
        samples_x = []
        samples_y = []
        for _ in range(n_samples):
            c = np.clip(dist.sample().detach().numpy(), exp.LOWER_CONTEXT_BOUNDS, exp.UPPER_CONTEXT_BOUNDS)
            if projection is not None:
                c = projection(c)
            samples_x.append(c[0])
            samples_y.append(c[1])

        ax.scatter(samples_x, samples_y, color=scalar_map.to_rgba(cv), alpha=alpha, marker=marker, s=5, edgecolors=None)

    if scatter_mean:
        ax.scatter(exp.TARGET_MEAN[0], exp.TARGET_MEAN[1], color="black", linewidth=2, marker="x", clip_on=False,
                   zorder=100)

    # Compute the bounding box of the context space
    xbounds = [exp.LOWER_CONTEXT_BOUNDS[0], exp.UPPER_CONTEXT_BOUNDS[0]]
    ybounds = [exp.LOWER_CONTEXT_BOUNDS[1], exp.UPPER_CONTEXT_BOUNDS[1]]
    vals = []
    for i in range(0, 2):
        for j in range(0, 2):
            x = np.array([xbounds[i], ybounds[j]])
            vals.append(projection(x) if projection is not None else x)
    lbs = np.min(vals, axis=0)
    ubs = np.max(vals, axis=0)

    ax.set_xlim([lbs[0], ubs[0]])
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=FONT_SIZE)

    ax.set_ylim([lbs[1], ubs[1]])
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE)

    if xticks is not None:
        ax.xaxis.set_ticks(xticks)
        ax.xaxis.set_ticklabels(xticks)

    if yticks is not None:
        ax.yaxis.set_ticks(yticks)
        if hide_y_ticks:
            ax.yaxis.set_ticklabels([""] * len(yticks))
        else:
            ax.yaxis.set_ticklabels(yticks)


def add_plots(exp, ax_top, axs_bottom, lines, labels, dist_vis=None, ignore=[]):
    ax0_limits = [-np.inf, np.inf]
    for method in METHODS:
        if method not in ignore:
            exp.curriculum = CurriculumType.from_string(method)
            exp.use_true_rew = "sac" in exp.get_log_dir() and method == "self_paced_v2"
            method_dir = os.path.dirname(exp.get_log_dir())
            if os.path.exists(method_dir):
                line = add_plot(method_dir, ax_top, COLOR_DICT[method], marker=MARKER_DICT[method],
                                markevery=MARKEVERY[exp.get_env_name()])
                if line is not None:
                    lines.append(line)
                    labels.append(LABEL_DICT[method])
                    cur_lim = ax_top.get_xlim()
                    ax0_limits = [np.maximum(ax0_limits[0], cur_lim[0]), np.minimum(ax0_limits[1], cur_lim[1])]
                if method == "self_paced":
                    if dist_vis is not None and dist_vis == exp.get_env_name():
                        for i, iter in enumerate(DIST_ITERS[dist_vis]):
                            visualize_distribution(exp, axs_bottom[i], iter, 0.5, [0., 0.2, 0.4],
                                                   xticks=DIST_XTICKS[dist_vis], yticks=DIST_YTICKS[dist_vis],
                                                   xlabel=DIST_XLABELS[dist_vis], hide_y_ticks=i > 0,
                                                   ylabel=DIST_YLABELS[dist_vis] if i == 0 else None,
                                                   seeds=DIST_SEEDS[dist_vis], n_samples=DIST_NSAMPLES[dist_vis],
                                                   projection=DIST_PROJECTIONS[dist_vis], markers=["o", "^", "s"],
                                                   scatter_mean=DIST_SHOW_MEAN[dist_vis])

    if ax0_limits[0] != -np.inf and ax0_limits[1] != np.inf:
        ax_top.set_xlim(ax0_limits)


def compute_indices(n_objects, n_grids, spacing):
    available_grids = n_grids - (n_objects - 1) * spacing
    grids_per_object = int(available_grids / n_objects)
    remainder = available_grids % n_objects

    indices = []
    count = 0
    for i in range(0, n_objects):
        end = count + grids_per_object
        if remainder > 0:
            end += 1
            remainder -= 1
        indices.append((count, end))
        count = end + spacing

    return indices


def main():
    global LABEL_DICT
    global COLOR_DICT
    global MARKER_DICT
    global METHODS

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_log_dir", type=str, default="logs")
    parser.add_argument("--env", type=str, default=["point_mass"], nargs="*",
                        choices=["point_mass", "point_mass_2d", "ball_catching"])
    parser.add_argument("--learner", type=str, default=["trpo"], nargs="*", choices=["trpo", "ppo", "sac"])
    parser.add_argument("--dist_vis", required=False, type=str)
    parser.add_argument("--methods", nargs="*", type=str,
                        choices=["self_paced", "self_paced_v2", "alp_gmm", "random", "default", "goal_gan"])

    args, remainder = parser.parse_known_args()
    parameters = parse_parameters(remainder)
    if len(args.env) != len(args.learner):
        raise RuntimeError("Number of envs and learners must be equal")

    if args.methods is not None and len(args.methods) != 0:
        METHODS = args.methods

    n_envs = len(args.env)
    if n_envs > 2:
        print("At most two envs are allowed!")

    f = plt.figure(figsize=(WIDTH, MUL * WIDTH))

    n_rows = 110
    n_cols = 300

    gs = f.add_gridspec(n_rows, n_cols)
    if n_envs == 1:
        axs_top = [f.add_subplot(gs[0:50, :])]
    else:
        axs_top = [f.add_subplot(gs[0:50, 0:140]), f.add_subplot(gs[0:50, 160:])]
    axs_bottom = [f.add_subplot(gs[77:105, idx[0]:idx[1]]) for idx in compute_indices(N_DIST_ITERS, n_cols, 11)]

    for i in range(0, len(axs_bottom)):
        axs_bottom[i].tick_params(axis='both', which='major', labelsize=FONT_SIZE)
        axs_bottom[i].tick_params(axis='both', which='minor', labelsize=FONT_SIZE)

    lines = []
    kl_lines = []
    labels = []
    kl_labels = []
    for k in range(0, len(args.env)):
        axs_top[k].tick_params(axis='both', which='major', labelsize=FONT_SIZE)
        axs_top[k].tick_params(axis='both', which='minor', labelsize=FONT_SIZE)
        axs_top[k].set_xlabel(r"Iteration", fontsize=FONT_SIZE)
        axs_top[k].set_title(ENV_NAMES[args.env[k]], fontsize=FONT_SIZE)

        if args.env[k] == "point_mass":
            from deep_sprl.experiments import PointMassExperiment
            exp = PointMassExperiment(args.base_log_dir, "default", args.learner[k], parameters, 1)
        elif args.env[k] == "point_mass_2d":
            from deep_sprl.experiments import PointMass2DExperiment
            exp = PointMass2DExperiment(args.base_log_dir, "default", args.learner[k], parameters, 1)
        else:
            from deep_sprl.experiments import BallCatchingExperiment
            exp = BallCatchingExperiment(args.base_log_dir, "default", args.learner[k], parameters, 1)

        new_lines = []
        new_kl_lines = []
        new_labels = []
        new_kl_labels = []
        if args.env[k] != "ball_catching":
            add_plots(exp, axs_top[k], axs_bottom, new_lines, new_labels, dist_vis=args.dist_vis)
            if args.env[k].startswith("point_mass"):
                add_sprl_plot(exp, axs_top[k], new_lines, new_labels, args.base_log_dir, COLOR_DICT["sprl"])
        else:
            LABEL_DICT["self_paced"] = r"SPDL*"
            LABEL_DICT["self_paced_v2"] = r"SPDL2*"
            LABEL_DICT["goal_gan"] = r"GoalGAN*"
            COLOR_DICT["self_paced"] = "C5"
            COLOR_DICT["self_paced_v2"] = "C6"
            COLOR_DICT["goal_gan"] = "C8"
            MARKER_DICT["self_paced"] = "v"
            MARKER_DICT["goal_gan"] = "d"
            add_plots(exp, axs_top[k], axs_bottom, new_lines, new_labels, dist_vis=None)

            exp = BallCatchingExperiment(args.base_log_dir, "default", args.learner[k],
                                         {**parameters, "INIT_CONTEXT": False}, 1)
            LABEL_DICT = {"self_paced": r"SPDL", "goal_gan": r"GoalGAN", "self_paced_v2": r"SPDL2"}
            COLOR_DICT = {"self_paced": "C0", "goal_gan": "C4", "self_paced_v2": "C1"}
            MARKER_DICT = {"self_paced": "^", "goal_gan": "D", "self_paced_v2": "x"}
            add_plots(exp, axs_top[k], axs_bottom, new_lines, new_labels, dist_vis="ball_catching")

            exp = BallCatchingExperiment(args.base_log_dir, "default", args.learner[k],
                                         {**parameters, "INIT_POLICY": False}, 1)
            LABEL_DICT = {"default": r"Default*"}
            COLOR_DICT = {"default": "C7"}
            MARKER_DICT = {"default": "."}
            add_plots(exp, axs_top[k], axs_bottom, new_lines, new_labels, dist_vis=None)

        # Only add new lines
        for new_line, new_label in zip(new_lines, new_labels):
            if new_label not in labels:
                lines.append(new_line)
                labels.append(new_label)

        for new_kl_line, new_kl_label in zip(new_kl_lines, new_kl_labels):
            if new_kl_label not in kl_labels:
                kl_lines.append(new_kl_line)
                kl_labels.append(new_kl_label)

        axs_top[k].grid()

    for i in range(0, len(axs_bottom)):
        axs_bottom[i].grid()

    lgd = f.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.48, 0.95), ncol=9, fontsize=FONT_SIZE,
                   handlelength=1.0, labelspacing=0., handletextpad=0.5, columnspacing=1.0)

    axs_top[0].set_ylabel(r"Reward", fontsize=FONT_SIZE)
    filename = ""
    for env, learner in zip(args.env, args.learner):
        if len(filename) == 0:
            filename += env + "_" + learner
        else:
            filename += "_" + env + "_" + learner
    key = "+".join(args.env)
    bbox = BBOXES[key] if key in BBOXES else None
    plt.savefig(filename + ".pdf", bbox_extra_artists=(lgd,), bbox_inches=bbox)


if __name__ == "__main__":
    main()
