# Self Paced Deep Reinforcement Learning

## Installation

It is easiest to setup a virtual or conda environment in order to isolate the packages installed for this project from your global python installation. We used Python 3.6.10 on Ubuntu 18.04 LTS for the experiments. You can easily install the required dependencies by executing
```bash
pip install -r requirements.txt
```

This will install all packages required to run the point mass experiments. If you furthermore want to run the ball catching experiment, you need to also execute
```
pip install -r requirements_ext.txt
```
This will install a wrapper for the MuJoCo simulation library. For this to work, you need to have set up MuJoCo according to [this guide](https://github.com/openai/mujoco-py).

There exist a convenience script for running the experiments: **run_experiments.sh**. The script takes one argument that specifies the seed with which the experiments will be run. So in order to run all experiments with seed 1, you need to execute
```bash
./run_experiments.sh 1
```

After running the experiments for the desired number of seeds, the results can be visualized using the following command
```bash
python visualize_results.py --env point_mass point_mass_2d --learner ppo ppo
python visualize_results.py --env point_mass point_mass_2d --learner trpo trpo
python visualize_results.py --env point_mass point_mass_2d --learner sac sac
python visualize_results.py --env ball_catching --learner ppo
python visualize_results.py --env ball_catching --learner trpo
python visualize_results.py --env ball_catching --learner sac
```

To visualize the context distributions for a set of seeds you can also execute the following commands.
```bash
python visualize_results.py --env point_mass point_mass_2d --learner ppo ppo --dist_vis point_mass_2d
python visualize_results.py --env point_mass point_mass_2d --learner trpo trpo --dist_vis point_mass_2d
python visualize_results.py --env point_mass point_mass_2d --learner sac sac --dist_vis point_mass_2d
python visualize_results.py --env ball_catching --learner ppo --dist_vis ball_catching
python visualize_results.py --env ball_catching --learner trpo --dist_vis ball_catching
python visualize_results.py --env ball_catching --learner sac --dist_vis ball_catching
```
Keep in mind that this requires a certain amount of seeds to be run (otherwise the script will return an error). You can also change the seeds that are visualized in the **visualize_results.py** script.
