python -u run_experiment.py --type default --seed $1
python -u run_experiment.py --type random --seed $1
python -u run_experiment.py --type self_paced --seed $1
python -u run_experiment.py --type alp_gmm --seed $1

python -u run_experiment.py --env point_mass_2d --type default --seed $1
python -u run_experiment.py --env point_mass_2d --type random --seed $1
python -u run_experiment.py --env point_mass_2d --type self_paced --seed $1
python -u run_experiment.py --env point_mass_2d --type alp_gmm --seed $1
