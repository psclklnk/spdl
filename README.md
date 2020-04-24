# Self Paced Deep Reinforcement Learning

## Installation

It is easiest to setup a virtual or conda environment in order to isolate the packages installed for this project from your global python installation. We used Python 3.6.10 on Ubuntu 18.04 LTS for the experiments. You can easily install the required dependencies by executing
```bash
pip install -r requirements.txt
```
Please not that this also installs the MuJoCo simulation library. For this to work, you need to have set up MuJoCo according to [this guide](https://github.com/openai/mujoco-py).

There exists a convenience script for running the experiments: **run_experiments.sh**. It will run all combinations of CL and RL algorithms in all environments. The script takes one argument that specifies the seed with which the experiments will be run. So in order to run all experiments with seed 1, you need to execute
```bash
./run_all_experiments.sh 1
```

After running the experiments for the desired number of seeds, the results can be visualized using the following command
```bash
python visualize_results.py --env point_mass --learner trpo
python visualize_results.py --env point_mass --learner ppo
python visualize_results.py --env point_mass --learner sac
python visualize_results.py --env point_mass_2d --learner trpo
python visualize_results.py --env point_mass_2d --learner ppo
python visualize_results.py --env point_mass_2d --learner sac
python visualize_results.py --env ball_catching --learner trpo
python visualize_results.py --env ball_catching --learner ppo
python visualize_results.py --env ball_catching --learner sac
```
