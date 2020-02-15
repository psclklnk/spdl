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

There exist two convenience scripts for running the experiments: **run_experiments.sh** and **run_all_experiments.sh**. The first script runs only the point mass experiments, while the latter in addition runs the ball catching experiment. Both scripts take one argument that specifies the seed with which the experiments will be run. So in order to run all experiments with seed 1, you need to execute
```bash
./run_all_experiments.sh 1
```

After running the experiments for the desired number of seeds, the results can be visualized using the following command
```bash
python visualize_results.py --env point_mass
python visualize_results.py --env point_mass_2d
python visualize_results.py --env ball_catching
```
