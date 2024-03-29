python -u run_experiment.py --env point_mass_2d --type self_paced --learner trpo --seed $1
python -u run_experiment.py --env point_mass_2d --type self_paced --learner ppo --seed $1
python -u run_experiment.py --env point_mass_2d --type self_paced --learner sac --seed $1
python -u run_experiment.py --env point_mass_2d --type self_paced_v2 --learner trpo --seed $1
python -u run_experiment.py --env point_mass_2d --type self_paced_v2 --learner ppo --seed $1
python -u run_experiment.py --env point_mass_2d --type self_paced_v2 --learner sac --seed $1 --true_rewards
python -u run_experiment.py --env point_mass_2d --type alp_gmm --learner trpo --seed $1
python -u run_experiment.py --env point_mass_2d --type alp_gmm --learner ppo --seed $1
python -u run_experiment.py --env point_mass_2d --type alp_gmm --learner sac --seed $1
python -u run_experiment.py --env point_mass_2d --type goal_gan --learner trpo --seed $1
python -u run_experiment.py --env point_mass_2d --type goal_gan --learner ppo --seed $1
python -u run_experiment.py --env point_mass_2d --type goal_gan --learner sac --seed $1
python -u run_experiment.py --env point_mass_2d --type default --learner trpo --seed $1
python -u run_experiment.py --env point_mass_2d --type default --learner ppo --seed $1
python -u run_experiment.py --env point_mass_2d --type default --learner sac --seed $1
python -u run_experiment.py --env point_mass_2d --type random --learner trpo --seed $1
python -u run_experiment.py --env point_mass_2d --type random --learner ppo --seed $1
python -u run_experiment.py --env point_mass_2d --type random --learner sac --seed $1
