python -u run_experiment.py --env point_mass --type self_paced --learner trpo --seed $1
python -u run_experiment.py --env point_mass --type self_paced --learner ppo --seed $1
python -u run_experiment.py --env point_mass --type self_paced --learner sac --seed $1
python -u run_experiment.py --env point_mass --type self_paced_v2 --learner trpo --seed $1
python -u run_experiment.py --env point_mass --type self_paced_v2 --learner ppo --seed $1
python -u run_experiment.py --env point_mass --type self_paced_v2 --learner sac --seed $1 --true_rewards
python -u run_experiment.py --env point_mass --type alp_gmm --learner trpo --seed $1
python -u run_experiment.py --env point_mass --type alp_gmm --learner ppo --seed $1
python -u run_experiment.py --env point_mass --type alp_gmm --learner sac --seed $1
python -u run_experiment.py --env point_mass --type goal_gan --learner trpo --seed $1
python -u run_experiment.py --env point_mass --type goal_gan --learner ppo --seed $1
python -u run_experiment.py --env point_mass --type goal_gan --learner sac --seed $1
python -u run_experiment.py --env point_mass --type default --learner trpo --seed $1
python -u run_experiment.py --env point_mass --type default --learner ppo --seed $1
python -u run_experiment.py --env point_mass --type default --learner sac --seed $1
python -u run_experiment.py --env point_mass --type random --learner trpo --seed $1
python -u run_experiment.py --env point_mass --type random --learner ppo --seed $1
python -u run_experiment.py --env point_mass --type random --learner sac --seed $1
