import torch
from utils.penalty_multipliers import real_corr_per_dataset_per_loss
from tqdm import tqdm
from helper import calculate_test_results, create_folder_if_it_doesnt_exist, get_x_test, set_seeds
import pandas as pd
import numpy as np

import os
os.chdir("./../")

base_results_path = 'results/real_data'

alpha = 0.1

def split_to_calibration_and_test(file):
    y_upper = file['test_y_upper'].item()
    y_lower = file['test_y_lower'].item()
    y = file['y_test'].item()

    y_upper = np.array(list(map(float, y_upper.strip('[]').split(','))))
    y_lower = np.array(list(map(float, y_lower.strip('[]').split(','))))
    y = np.array(list(map(float, y.strip('[]').split(','))))

    idx = np.arange(len(y_upper))  # np.random.permutation(len(y_upper))
    n_half = int(np.floor(len(y_upper) / 2))
    idx_test, idx_cal = idx[:n_half], idx[n_half:2 * n_half]

    cal_y_upper = y_upper[idx_cal]
    cal_y_lower = y_lower[idx_cal]
    cal_y = y[idx_cal]

    test_y_upper = y_upper[idx_test]
    test_y_lower = y_lower[idx_test]
    test_y = y[idx_test]

    return cal_y, cal_y_upper, cal_y_lower, test_y, test_y_upper, test_y_lower, idx_test, idx_cal


def calibrate_test(cal_y, cal_y_upper, cal_y_lower, test_y_upper, test_y_lower):
    error_low = cal_y_lower - cal_y
    error_high = cal_y - cal_y_upper
    err = np.maximum(error_high, error_low)
    err = np.sort(err)
    index = int(np.ceil((1 - alpha) * (err.shape[0] + 1))) - 1
    index = min(max(index, 0), err.shape[0] - 1)
    q = err[index]
    return q, test_y_upper + q, test_y_lower - q


def calculate_calibrated_results(dataset_name, corr_mult, seed, method):
    set_seeds(seed)
    if method == 'qr':
        method = f"loss=batch_qr_bs=1024_corr_mult={corr_mult}_hsic_mult=0.0"
    file = pd.read_csv(
        f"{base_results_path}/{dataset_name}/{method}/seed={seed}.csv")
    cal_y, cal_y_upper, cal_y_lower, test_y, test_y_upper, test_y_lower, idx_test, idx_cal = \
        split_to_calibration_and_test(file)
    x_test = get_x_test(dataset_name, seed)[idx_test]

    # getting the calibrated y_upper, y_lower
    q, test_y_upper, test_y_lower = calibrate_test(cal_y, cal_y_upper, cal_y_lower, test_y_upper, test_y_lower)

    return_values = calculate_test_results(x_test, torch.Tensor(test_y),
                                           torch.Tensor(test_y_upper), torch.Tensor(test_y_lower))
    return_values['q'] = q

    return_values['test_y_upper'] = [list(test_y_upper)]
    return_values['test_y_lower'] = [list(test_y_lower)]
    return_values['y_test'] = [list(test_y)]

    for feature in ['train_chi2_pvalue', 'train_chi2_statistic', 'train_pearson_corr', 'train_pearson_pvalue',
                    'train_coverage', 'train_interval_len', 'train_interval_len_var']:
        return_values[feature] = [0]
    return return_values


if __name__ == '__main__':
    dataset_names = ['kin8nm', 'naval', 'meps_19', 'meps_20', 'meps_21',
                     'facebook_1', 'facebook_2', 'blog_data', 'bio']

    for dataset_name in dataset_names:
        print(f"dataset: {dataset_name}")
        for corr_mult in [0., real_corr_per_dataset_per_loss['qr'][dataset_name]]:
            for seed in tqdm(range(0, 3)):
                return_values = calculate_calibrated_results(dataset_name, corr_mult, seed, method='qr')
                save_dir = f"{base_results_path}/{dataset_name}/cal_loss=batch_qr_bs=1024_corr_mult={corr_mult}_hsic_mult=0.0"
                create_folder_if_it_doesnt_exist(save_dir)
                pd.DataFrame(return_values).to_csv(f"{save_dir}/seed={seed}.csv", index=[seed])


    # for dataset_name in ['meps_19']:
    #     print(f"dataset: {dataset_name}")
    #     for seed in tqdm(range(0, 30)):
    #         return_values = calculate_calibrated_results(dataset_name, corr_mult=None, seed=seed, method='qr_forest')
    #         save_dir = f"{base_results_path}/{dataset_name}/cal_qr_forest"
    #         create_folder_if_it_doesnt_exist(save_dir)
    #         pd.DataFrame(return_values).to_csv(f"{save_dir}/seed={seed}.csv", index=[seed])


