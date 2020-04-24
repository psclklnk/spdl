python -u run_experiment.py --env ball_catching --type self_paced --learner trpo --seed $1
python -u run_experiment.py --env ball_catching --type self_paced --learner ppo --seed $1
python -u run_experiment.py --env ball_catching --type self_paced --learner sac --seed $1
python -u run_experiment.py --env ball_catching --type goal_gan --learner trpo --seed $1
python -u run_experiment.py --env ball_catching --type goal_gan --learner ppo --seed $1
python -u run_experiment.py --env ball_catching --type goal_gan --learner sac --seed $1
python -u run_experiment.py --env ball_catching --type alp_gmm --learner trpo --seed $1
python -u run_experiment.py --env ball_catching --type alp_gmm --learner ppo --seed $1
python -u run_experiment.py --env ball_catching --type alp_gmm --learner sac --seed $1
python -u run_experiment.py --env ball_catching --type default --learner trpo --seed $1
python -u run_experiment.py --env ball_catching --type default --learner ppo --seed $1
python -u run_experiment.py --env ball_catching --type default --learner sac --seed $1

python -u run_experiment.py --env ball_catching --type self_paced --learner trpo --seed $1 --INIT_CONTEXT False
python -u run_experiment.py --env ball_catching --type self_paced --learner ppo --seed $1 --INIT_CONTEXT False
python -u run_experiment.py --env ball_catching --type self_paced --learner sac --seed $1 --INIT_CONTEXT False
python -u run_experiment.py --env ball_catching --type goal_gan --learner trpo --seed $1 --INIT_CONTEXT False
python -u run_experiment.py --env ball_catching --type goal_gan --learner ppo --seed $1 --INIT_CONTEXT False
python -u run_experiment.py --env ball_catching --type goal_gan --learner sac --seed $1 --INIT_CONTEXT False

python -u run_experiment.py --env ball_catching --type default --learner trpo --seed $1 --INIT_POLICY False
python -u run_experiment.py --env ball_catching --type default --learner ppo --seed $1 --INIT_POLICY False
python -u run_experiment.py --env ball_catching --type default --learner sac --seed $1 --INIT_POLICY False