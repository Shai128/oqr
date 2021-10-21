import subprocess
from itertools import product
import time
import sys
sys.path.append("./../")

from reproducible_experiments.run_experiment import run_experiment

def cartesian_product(inp):
    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))

processes_to_run_in_parallel = 2

all_params = {
    'loss': ['batch_wqr'],
    'corr_mult': [0],
    'hsic_mult': [0],
    'data_type': ['SYN'],
    'seed': [(0,15), (15, 30)],
    'data': ['3', '10'],
    'method': ['QR']

}

"""
POSSIBLE_REAL_DATA_NAMES = ['kin8nm', 'naval', 'meps_19', 'meps_20', 'meps_21', 'facebook_1', 'facebook_2',
'blog_data', 'bio']
"""


if 'seed' in all_params and len(all_params['seed']) == 0:
    del all_params['seed']

has_seed = 'seed' in all_params

params = list(cartesian_product(all_params))




processes_to_run_in_parallel = min(processes_to_run_in_parallel, len(params))

corr_mults = {'qr': {
    'kin8nm': 0.5,
    'naval': 0.1,
    'meps_19': 0.5,
    'meps_20': 0.5,
    'meps_21': 0.5,
    'facebook_1': 0.5,
    'facebook_2': 0.5,
    'blog_data': 0.5,
    'bio': 0.1
},
    'int': {
        'kin8nm': 0.5,
        'naval': 0.1,
        'meps_19': 3,
        'meps_20': 3,
        'meps_21': 3,
        'facebook_1': 0.1,
        'facebook_2': 0.1,
        'blog_data': 1.,
        'bio': 0.1
    }}



if __name__ == '__main__':
    print("jobs to do: ", len(params))
    # initializing proccesses_to_run_in_parallel workers
    workers = []
    jobs_finished_so_far = 0
    assert len(params) >= processes_to_run_in_parallel
    for _ in range(processes_to_run_in_parallel):
        curr_params = params.pop(0)
        p = run_experiment(curr_params)
        workers.append(p)

    # creating a new process when an old one dies
    while len(params) > 0:
        dead_workers_indexes = [i for i in range(len(workers)) if (workers[i].poll() is not None)]
        for i in dead_workers_indexes:
            worker = workers[i]
            worker.communicate()
            jobs_finished_so_far += 1
            if len(params) > 0:
                curr_params = params.pop(0)
                p = run_experiment(curr_params)
                workers[i] = p
                if jobs_finished_so_far % processes_to_run_in_parallel == 0:
                    print(f"finished so far: {jobs_finished_so_far}, {len(params)} jobs left")
        if has_seed:
            time.sleep(10)
        else:
            time.sleep(20)

    # joining all last proccesses
    for worker in workers:
        worker.communicate()
        jobs_finished_so_far += 1

    print("finished all")
