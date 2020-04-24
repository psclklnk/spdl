import os
import sys
import argparse
from gym.wrappers import Monitor

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

from deep_sprl.experiments import BallCatchingExperiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_log_dir", type=str, default="logs")
    parser.add_argument("--type", type=str, default="default",
                        choices=["default", "random", "self_paced", "goal_gan", "alp_gmm"])
    parser.add_argument("--learner", type=str, required=True, choices=["trpo", "ppo", "sac"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--video_dir", type=str, default="video")

    args = parser.parse_args()

    video_dir = os.path.join(os.path.dirname(__file__), "..", args.video_dir, "ball_catching", args.type,
                             "seed-" + str(args.seed))
    os.makedirs(video_dir, exist_ok=True)

    exp = BallCatchingExperiment(args.base_log_dir, args.type, args.learner, {}, args.seed)
    log_dir = os.path.join(os.path.dirname(__file__), "..", exp.get_log_dir(), "iteration-" + str(495))
    monitor = Monitor(exp.eval_env, video_dir, force=True, video_callable=lambda episode_id: True)

    model = exp.learner.load_for_evaluation(os.path.join(log_dir, "model"), exp.eval_env)
    for i in range(0, 10):
        obs = monitor.reset()
        done = False
        while not done:
            action = model.step(obs, state=None, deterministic=False)
            obs, rewards, done, infos = monitor.step(action)

    monitor.close()


if __name__ == "__main__":
    main()
