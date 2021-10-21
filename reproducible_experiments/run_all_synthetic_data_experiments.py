###############################################################################
# Script for reproducing the results of OQR paper
###############################################################################
import sys

sys.path.insert(1, '..')

import time
from reproducible_experiments.run_experiment import run_experiment
from utils.penalty_multipliers import syn_corr_per_dataset_per_loss

processes_to_run_in_parallel = 1

loss_functions = ['batch_qr', 'batch_int']
datasets = ['3', '10']

corr_mults = syn_corr_per_dataset_per_loss

all_params = []
seed = (0, 30)
# adding the during_training configurations

for dataset in datasets:
    all_params += [  # during training results of vanilla QR
        {
            'loss': 'batch_qr',
            'data': dataset,
            'data_type': 'SYNTHETIC',
            'seed': 42,
            'corr_mult': 0,
            'hsic_mult': 0,
            'save_training_results': True,
            'method': 'QR'
        },
        {   # during training results of OQR
            'loss': 'batch_qr',
            'data': dataset,
            'data_type': 'SYNTHETIC',
            'seed': 42,
            'corr_mult': corr_mults['qr'][dataset],
            'hsic_mult': 0,
            'save_training_results': True,
            'method': 'QR'

        },
        {  # WQR results
            'loss': 'batch_qr',
            'data': dataset,
            'data_type': 'SYNTHETIC',
            'seed': (0, 30),
            'corr_mult': 0,
            'hsic_mult': 0,
            'save_training_results': True,
            'method': 'QR'
        },
        {   # OWQR results
            'loss': 'batch_qr',
            'data': dataset,
            'data_type': 'SYNTHETIC',
            'seed': (0, 30),
            'corr_mult': corr_mults['wqr'][dataset],
            'hsic_mult': 0,
            'save_training_results': True,
            'method': 'QR'
        }
    ]

# adding to a list all running configurations
for data in datasets:
    for loss in loss_functions:
        all_params += [
            {
                'loss': loss,
                'data': data,
                'data_type': 'SYNTHETIC',
                'seed': seed,
                'corr_mult': 0,
                'hsic_mult': 0,
                'method': 'QR'

            },
            {
                'loss': loss,
                'data': data,
                'data_type': 'SYNTHETIC',
                'seed': seed,
                'corr_mult': corr_mults[loss.replace("batch_", "")][data],
                'hsic_mult': 0,
                'method': 'QR'

            }]

processes_to_run_in_parallel = min(processes_to_run_in_parallel, len(all_params))

if __name__ == '__main__':
    print("jobs to do: ", len(all_params))

    # initializing the first workers
    workers = []
    jobs_finished_so_far = 0
    assert len(all_params) >= processes_to_run_in_parallel
    for _ in range(processes_to_run_in_parallel):
        curr_params = all_params.pop(0)
        p = run_experiment(curr_params)
        workers.append(p)

    # creating a new process when an old one dies
    while len(all_params) > 0:
        dead_workers_indexes = [i for i in range(len(workers)) if (workers[i].poll() is not None)]
        for i in dead_workers_indexes:
            worker = workers[i]
            worker.communicate()
            jobs_finished_so_far += 1
            if len(all_params) > 0:
                curr_params = all_params.pop(0)
                p = run_experiment(curr_params)
                workers[i] = p
                if jobs_finished_so_far % processes_to_run_in_parallel == 0:
                    print(f"finished so far: {jobs_finished_so_far}, {len(all_params)} jobs left")

        time.sleep(5)

    # joining all last processes
    for worker in workers:
        worker.communicate()
        jobs_finished_so_far += 1

    print("finished all")
