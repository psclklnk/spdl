python -u run_experiment.py --env point_mass --type self_paced --seed $1
python -u run_experiment.py --env point_mass --type alp_gmm --seed $1
python -u run_experiment.py --env point_mass --type default --seed $1
python -u run_experiment.py --env point_mass --type random --seed $1

python -u run_experiment.py --env point_mass_2d --type self_paced --seed $1
python -u run_experiment.py --env point_mass_2d --type alp_gmm --seed $1
python -u run_experiment.py --env point_mass_2d --type default --seed $1
python -u run_experiment.py --env point_mass_2d --type random --seed $1

python -u run_experiment.py --env ball_catching --type self_paced --seed $1
python -u run_experiment.py --env ball_catching --type goal_gan --seed $1
python -u run_experiment.py --env ball_catching --type default --seed $1
