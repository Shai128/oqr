import subprocess
import os

def run_experiment(experiment_params):

    data_type = experiment_params["data_type"]

    # & for windows or ; for mac
    if os.name == 'nt':
        separate = '&'
    else:
        separate = ';'

    command = f'cd .. {separate} python main.py --bs 1024 --ds_type {data_type}'

    if 'syn' in data_type.lower():
        command += ' --test_ratio 0.2 '
    else:
        assert 'real' in data_type.lower()
        command += ' --test_ratio 0.4 --nl 3 --dropout 0.1 '

    for param in ['hsic_mult', 'corr_mult', 'data', 'loss', 'method']:
        if param in experiment_params:
            command += f' --{param} {experiment_params[param]} '

    if 'seed' in experiment_params:
        if type(experiment_params["seed"]) == tuple:
            seed_begin, seed_end = experiment_params["seed"]
            seed_param = f' --seed_begin {seed_begin} --seed_end {seed_end} '
        else:
            seed = experiment_params["seed"]
            seed_param = f' --seed {seed} '
    else:
        seed_param = ''

    command += seed_param

    if 'save_training_results' in experiment_params and experiment_params['save_training_results']:
        command += ' --save_training_results 1'

    process = subprocess.Popen(command, shell=True)

    return process