###############################################################################
# Script for reproducing the results of OQR paper
###############################################################################

import time
from reproducible_experiments.run_experiment import run_experiment
from utils.penalty_multipliers import real_corr_per_dataset_per_loss
processes_to_run_in_parallel = 1

loss_functions = ['batch_qr', 'batch_int', 'batch_wqr']
datasets = ['kin8nm', 'naval', 'meps_19', 'meps_20', 'meps_21', 'facebook_1', 'facebook_2',
            'blog_data', 'bio']


corr_mults = real_corr_per_dataset_per_loss
hsic_mults = corr_mults['hsic_qr']
seed = (3, 10)
# adding to a list all running configurations
all_params = []
for data in datasets:
    for loss in loss_functions:
        all_params += [
                {
                'loss': loss,
                'data': data,
                'data_type': 'REAL',
                'seed': seed,
                'corr_mult': 0,
                'hsic_mult': 0,
                'method': 'QR'
            },
            {
                'loss': loss,
                'data': data,
                'data_type': 'REAL',
                'seed': seed,
                'corr_mult': corr_mults[loss.replace("batch_", "")][data],
                'hsic_mult': 0,
                'method': 'QR'

            }]
    all_params += [{
                'loss': 'batch_qr',
                'data': data,
                'data_type': 'REAL',
                'seed': seed,
                'corr_mult': 0,
                'hsic_mult': hsic_mults[data],
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
