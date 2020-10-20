import os
import sys
import argparse
from PIL import Image

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

from deep_sprl.experiments import BallCatchingExperiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_log_dir", type=str, default="logs")
    parser.add_argument("--learner", type=str, default="trpo", choices=["trpo", "ppo", "sac"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--picture_dir", type=str, default="pictures")
    parser.add_argument("--init_context", action="store_true")

    args = parser.parse_args()

    if args.init_context:
        exp = BallCatchingExperiment(args.base_log_dir, "self_paced", args.learner, {"VISUALIZE": True}, args.seed)
    else:
        exp = BallCatchingExperiment(args.base_log_dir, "self_paced", args.learner,
                                     {"INIT_CONTEXT": False, "VISUALIZE": True}, args.seed)
    images = []
    for iter in range(0, 500, 5):
        exp.eval_env.reset()

        teacher = exp.create_self_paced_teacher()
        teacher.load(os.path.join(os.path.dirname(__file__), "..", exp.get_log_dir(), "iteration-" + str(iter),
                                  "context_dist.npy"))

        contexts = [teacher.sample() for _ in range(30)]
        exp.eval_env.env.unwrapped.set_contexts(contexts)
        img = exp.eval_env.render(mode="rgb_array")
        images.append(Image.fromarray(img))

    picture_base = os.path.join(os.path.dirname(__file__), "..", args.picture_dir)
    os.makedirs(picture_base, exist_ok=True)
    images[0].save(os.path.join(picture_base, "ball_catching_curriculum_%s_%d.gif" % (args.learner, args.seed)),
                   format='GIF', save_all=True, append_images=images[1:], duration=100, loop=0)

    exp.eval_env.close()


if __name__ == "__main__":
    main()
