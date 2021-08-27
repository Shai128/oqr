import sys, os
import torch
import numpy as np
from scipy import stats
from skgarden import RandomForestQuantileRegressor
from torch.utils.data import TensorDataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
from datasets.datasets import scale_data, GetDataset
import pandas as pd

from utils.q_model_ens import MSEModel

results_path = './results/'
syn_data_path = './datasets/synthetic_data/'
dataset_base_path = "./datasets/real_data/"
REAL_DATA = 'real data'
SYN_DATA = 'synthetic data'


def get_wqr_weights(loss, x_tr, y_tr, x_va, y_va, args):
    if 'wqr' in loss:
        mse_model = MSEModel(in_dim=x_tr.shape[1], device=args.device)
        mse_model.fit(x_tr, y_tr, x_va, y_va)
        # w_tr = ((unscaled_x_tr[:, 0] == 0) + (unscaled_x_tr[:, 0] == 1) * 4).float()
        w_tr = abs(mse_model(x_tr) - y_tr.squeeze()) ** (1)  # torch.ones(len(x_tr))
        w_tr = w_tr.detach()
        # w_te = (mse_model(x_te) - y_te) ** (-2)
        # w_va = ((unscaled_x_va[:, 0] == 0) + (unscaled_x_va[:, 0] == 1) * 4).float()
        w_va = abs(mse_model(x_va) - y_va.squeeze()) ** (1)  # torch.ones(len(x_va))
        w_va = w_va.detach()
        w_tr = w_tr.to(args.device)
        w_va = w_va.to(args.device)

        def get_tr_weights(idx):
            return w_tr[idx]

    else:
        w_tr = w_va = None

        def get_tr_weights(idx):
            return None

    return w_tr, w_va, get_tr_weights


class IndexedDataset(TensorDataset):
    def __init__(self, *tensors: torch.Tensor) -> None:
        assert all(
            tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        tensors = tuple(tensor[index] for tensor in self.tensors)
        return tensors + (index,)

    def __len__(self):
        return self.tensors[0].size(0)


def save_results(dataset_name, data_type, x_train, unscaled_x_train, x_test, unscaled_x_test, y_train, y_test,
                 y_upper,
                 y_lower,
                 train_y_upper,
                 train_y_lower,
                 seed,
                 args,
                 minority_group_uncertainty=None,
                 group_feature=None):
    if data_type == SYN_DATA:
        synthetic_data_params = {
            'consider_groups': True,
            'group_feature': group_feature
        }
        x_train = unscaled_x_train
        x_test = unscaled_x_test
    else:
        synthetic_data_params = {
            'consider_groups': False
        }

    return_values = calculate_results(x_train,
                                      y_train,
                                      x_test,
                                      y_test,
                                      y_upper,
                                      y_lower,
                                      train_y_upper,
                                      train_y_lower,
                                      **synthetic_data_params)

    if data_type == SYN_DATA:
        dataset_name = f"minority_group_uncertainty={minority_group_uncertainty}"

    results_dir = get_method_results_dir(data_type, dataset_name, args)
    create_folder_if_it_doesnt_exist(results_dir)
    results_path = f"{results_dir}/seed={seed}.csv"

    pd.DataFrame(return_values, index=[seed]).to_csv(results_path)


def args_to_txt(args):
    if args.method == 'wqr':
        args_summary = 'wqr'
    elif args.method == 'qr_forest':
        args_summary = 'qr_forest'
    else:
        args_summary = str("loss=" + args.loss + "_bs=" + str(args.bs) + "_corr_mult=" +
                           str(args.corr_mult) + '_hsic_mult=' + str(args.hsic_mult))
    return args_summary


def get_method_results_dir(dataset_type, data_name, args):
    data_type_dir = 'syn_data' if dataset_type == SYN_DATA else 'real_data'
    return f"{results_path}{data_type_dir}/{data_name}/{args_to_txt(args)}"


def get_feature_as_function_of_mult_figure_dir(dataset_type, data_name, mult, args):
    data_type_dir = 'syn_data' if dataset_type == SYN_DATA else 'real_data'
    args_txt = f'loss={args.loss}_bs={args.bs}_mult={mult}'
    return f"{results_path}{data_type_dir}/{data_name}/{args_txt}"


def create_folder_if_it_doesnt_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def pearsons_corr(x, y):
    """
    computes the correlation between to samples of empirical samples
    Parameters
    ----------
    x - a vector if n samples drawn from X
    y - a vector if n samples drawn from Y

    Returns
    -------
    The empirical correlation between X and Y
    """
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))


def pearsons_corr2d(x, y):
    """

    Parameters
    ----------
    x - matrix of size n_vectors x n_elements
    y - matrix of size n_vectors x n_elements

    Returns
    -------
    A vector of size n_vectors where the i-th element is the correlation between x[i, :], y[i,:]
    """
    vx = x - torch.mean(x, dim=1).reshape(len(x), 1)
    vy = y - torch.mean(y, dim=1).reshape(len(x), 1)

    return torch.sum(vx * vy, dim=1) / (torch.sqrt(torch.sum(vx ** 2, dim=1)) * torch.sqrt(torch.sum(vy ** 2, dim=1)))


def pairwise_distances(x):
    # x should be two dimensional
    instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def GaussianKernelMatrix(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ / sigma)


def HSIC(x, y, s_x=1, s_y=1):
    m, _ = x.shape  # batch size
    K = GaussianKernelMatrix(x, s_x).float()
    L = GaussianKernelMatrix(y, s_y).float()
    H = torch.eye(m) - 1.0 / m * torch.ones((m, m))
    H = H.float().to(K.device)
    HSIC = torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H)))) / ((m - 1) ** 2)
    return HSIC


tanh = torch.nn.Tanh()


def compute_coverages_and_avg_interval_len(y, y_lower, y_upper):
    """
    computes prediction interval length and the coverage indicators.
    Parameters
    ----------
    y - n response variables
    y_lower - n upper bounds of the intervals that should contain y with a pre-specified probability
    y_upper - n lower bounds of the intervals that should contain y with a pre-specified probability

    Returns
    -------
    interval_lengths - a vector of size n that contains the length of each interval
    coverage_indicators - a vector of size n that contains a differential indicator that equals (approximately) 1
                            if the interval contains y, and (approximately) 0, otherwise.
    """

    interval_lengths = (y_upper - y_lower)

    coverage_indicators = tanh(50 * torch.min(y - y_lower, y_upper - y))

    coverage_indicators = (coverage_indicators + 1) / 2

    return coverage_indicators, interval_lengths


def chi_squared_test(y, y_lower, y_upper):
    # return np_histogram_chi_squared_test(y, y_lower, y_upper)
    n = len(y)

    coverage = ((y >= y_lower) & (y <= y_upper)).float()  # .type(torch.float64)
    interval_sizes = torch.Tensor(y_upper - y_lower)
    sorted_l, indices = interval_sizes.sort()

    alpha = 0.05
    indices = indices[int(n * (alpha / 2)):int(n * (1 - alpha / 2))]

    coverage_and_interval_sizes = torch.zeros((len(indices), 2))
    coverage_and_interval_sizes[:, 0] = coverage[indices]
    coverage_and_interval_sizes[:, 1] = interval_sizes[indices]

    statistic = chi2_pvalue = None
    number_of_bins = 20
    # min_bin_size = len(coverage)//number_of_bins
    min_bin_size = n // number_of_bins  # min(n // 5, int(6 // (len(coverage[coverage == 0]) / len(coverage))) * 2)
    while True:
        try:
            split = coverage_and_interval_sizes.split(min_bin_size)
            # split = split[:-1]
            obs_list = list(map(lambda t: torch.stack((t[:, 0].sum(), len(t[:, 0]) - t[:, 0].sum())).T, split))
            obs = torch.stack(obs_list, dim=0)
            statistic, chi2_pvalue, _, _ = stats.chi2_contingency(obs)
            break
        except:
            if min_bin_size == n:
                break
            min_bin_size = min(min_bin_size + 2, n)
    return statistic, chi2_pvalue


def wsc(X, y, y_upper, y_lower, delta=0.1, M=1000, verbose=False):
    def wsc_v(X, y, y_upper, y_lower, delta, v):
        n = len(y)
        cover = ((y >= y_lower) & (y <= y_upper)).astype(np.float32)
        z = np.dot(X, v)
        # Compute mass
        z_order = np.argsort(z)
        z_sorted = z[z_order]
        cover_ordered = cover[z_order]
        ai_max = int(np.round((1.0 - delta) * n))
        ai_best = 0
        bi_best = n
        cover_min = 1
        for ai in np.arange(0, ai_max):
            bi_min = np.minimum(ai + int(np.round(delta * n)), n)
            coverage = np.cumsum(cover_ordered[ai:n]) / np.arange(1, n - ai + 1)
            coverage[np.arange(0, bi_min - ai)] = 1
            bi_star = ai + np.argmin(coverage)
            cover_star = coverage[bi_star - ai]
            if cover_star < cover_min:
                ai_best = ai
                bi_best = bi_star
                cover_min = cover_star
        return cover_min, z_sorted[ai_best], z_sorted[bi_best]

    def sample_sphere(n, p):
        v = np.random.randn(p, n)
        v /= np.linalg.norm(v, axis=0)
        return v.T

    V = sample_sphere(M, p=X.shape[1])
    wsc_list = [[]] * M
    a_list = [[]] * M
    b_list = [[]] * M
    if verbose:
        for m in tqdm(range(M)):
            wsc_list[m], a_list[m], b_list[m] = wsc_v(X, y, y_upper, y_lower, delta, V[m])
    else:
        for m in range(M):
            wsc_list[m], a_list[m], b_list[m] = wsc_v(X, y, y_upper, y_lower, delta, V[m])

    idx_star = np.argmin(np.array(wsc_list))
    a_star = a_list[idx_star]
    b_star = b_list[idx_star]
    v_star = V[idx_star]
    wsc_star = wsc_list[idx_star]
    return wsc_star, v_star, a_star, b_star


def wsc_unbiased(X, y, y_upper, y_lower, delta=0.1, M=1000, test_size=0.75, random_state=2021, verbose=False):
    X, y, y_upper, y_lower = X.numpy(), y.numpy(), y_upper.numpy(), y_lower.numpy()

    def wsc_vab(X, y, y_upper, y_lower, v, a, b):
        cover = ((y >= y_lower) & (y <= y_upper)).astype(np.float32)
        z = np.dot(X, v)
        idx = np.where((z >= a) * (z <= b))
        coverage = np.mean(cover[idx])
        return coverage

    X_train, X_test, y_train, y_test, y_upper_train, y_upper_test, y_lower_train, y_lower_test = \
        train_test_split(X, y, y_upper, y_lower, test_size=test_size,
                         random_state=random_state)
    # Find adversarial parameters
    wsc_star, v_star, a_star, b_star = wsc(X_train, y_train, y_upper_train, y_lower_train, delta=delta, M=M,
                                           verbose=verbose)
    # Estimate coverage
    coverage = wsc_vab(X_test, y_test, y_upper_test, y_lower_test, v_star, a_star, b_star)
    return coverage


def run_tree_experiment(x_train, y_train, x_test, y_test, unscaled_x_train, unscaled_x_test, data_type,
                        minority_group_uncertainty, group_feature, args, s, d):
    rfqr = RandomForestQuantileRegressor(
        random_state=s, min_samples_leaf=40, n_estimators=200, n_jobs=-1)
    rfqr.fit(x_train.cpu(), y_train.cpu().flatten())
    y_upper = torch.Tensor(rfqr.predict(x_test.cpu(), quantile=95)).to(args.device)
    y_lower = torch.Tensor(rfqr.predict(x_test.cpu(), quantile=5)).to(args.device)
    train_y_upper = torch.Tensor(rfqr.predict(x_train.cpu(), quantile=95)).to(args.device)
    train_y_lower = torch.Tensor(rfqr.predict(x_train.cpu(), quantile=5)).to(args.device)
    save_results(d, data_type, x_train, unscaled_x_train, x_test, unscaled_x_test, y_train.squeeze(),
                 y_test.squeeze(),
                 y_upper,
                 y_lower,
                 train_y_upper,
                 train_y_lower,
                 s,
                 args,
                 minority_group_uncertainty=minority_group_uncertainty,
                 group_feature=group_feature)


def get_x_test(dataset_name, seed):
    set_seeds(seed)

    x, y = GetDataset(dataset_name, dataset_base_path)
    idx = np.random.permutation(len(x))
    x = x[idx]
    y = y[idx]
    y = y.reshape(-1, 1)

    data_out = scale_data(x, y, seed, test_size=0.4)

    x_tr, x_va, x_te, y_tr, y_va, y_te, y_al = \
        data_out.x_tr, data_out.x_va, data_out.x_te, data_out.y_tr, \
        data_out.y_va, data_out.y_te, data_out.y_al
    return x_te


def calculate_test_results(x_test, y_test, y_upper, y_lower):
    x_test = x_test.clone().detach()
    y_test = y_test.clone().detach()
    y_upper = y_upper.clone().detach()
    y_lower = y_lower.clone().detach()
    return_values = {}
    device = "cpu"
    coverage = ((y_test >= y_lower) & (y_test <= y_upper)).float()
    interval_sizes = (y_upper - y_lower).float()
    return_values['coverage'] = coverage.mean().item()
    return_values['interval_len'] = interval_sizes.mean().item()
    return_values['interval_len_var'] = interval_sizes.var().item()

    chi2_statistic, chi2_pvalue = chi_squared_test(y_test, y_lower, y_upper)
    return_values['test_chi2_pvalue'] = chi2_pvalue
    return_values['test_chi2_statistic'] = chi2_statistic

    x, y = coverage, interval_sizes
    test_pearson_corr, test_pearson_pvalue = stats.pearsonr(x.cpu().detach().numpy(), y.cpu().detach().numpy())
    return_values['test_pearson_corr'] = test_pearson_corr
    return_values['test_pearson_pvalue'] = test_pearson_pvalue

    x, y = coverage, interval_sizes
    idx = np.random.permutation(len(x))[:5000]
    hsic = HSIC(x[idx].reshape(len(idx), 1).to(device), y[idx].reshape(len(idx), 1).to(device)).item()
    return_values['test_hsic'] = hsic

    return_values['test_wsc'] = wsc_unbiased(x_test, y_test, y_upper, y_lower, M=500)
    return_values['test_wsc_diff'] = abs(float(return_values['test_wsc']) - return_values['coverage'])

    return return_values


def calculate_results(x_train, y_train, x_test, y_test, test_y_upper, test_y_lower, train_y_upper, train_y_lower,
                      consider_groups=True, group_feature=0):
    return_values = {}
    device = "cpu"
    x_train = torch.tensor(x_train).to(device)
    y_train = torch.tensor(y_train).to(device)
    x_test = torch.tensor(x_test).to(device)
    y_test = torch.tensor(y_test).to(device)
    test_y_upper = torch.tensor(test_y_upper).to(device)
    test_y_lower = torch.tensor(test_y_lower).to(device)
    train_y_upper = torch.tensor(train_y_upper).to(device)
    train_y_lower = torch.tensor(train_y_lower).to(device)

    y_upper = test_y_upper
    y_lower = test_y_lower

    return_values['test_y_upper'] = [list(test_y_upper.numpy())]
    return_values['test_y_lower'] = [list(test_y_lower.numpy())]
    return_values['y_test'] = [list(y_test.numpy())]

    return_values = {**return_values, **calculate_test_results(x_test, y_test, y_upper, y_lower)}

    train_coverage = torch.tensor((y_train >= train_y_lower) & (y_train <= train_y_upper), dtype=torch.float64).float()
    train_interval_sizes = torch.tensor(train_y_upper - train_y_lower, dtype=torch.float64).float()

    chi2_statistic, chi2_pvalue = chi_squared_test(y_train, train_y_lower, train_y_upper)
    return_values['train_chi2_pvalue'] = chi2_pvalue
    return_values['train_chi2_statistic'] = chi2_statistic

    x, y = train_coverage, train_interval_sizes
    train_pearson_corr, train_pearson_pvalue = stats.pearsonr(x.cpu().detach().numpy(), y.cpu().detach().numpy())
    return_values['train_pearson_corr'] = train_pearson_corr
    return_values['train_pearson_pvalue'] = train_pearson_pvalue

    y_upper = train_y_upper
    y_lower = train_y_lower
    coverage = torch.tensor((y_train >= y_lower) & (y_train <= y_upper), dtype=torch.float64).float()
    interval_sizes = torch.tensor(y_upper - y_lower, dtype=torch.float64).float()
    return_values['train_coverage'] = coverage.mean().item()
    return_values['train_interval_len'] = interval_sizes.mean().item()
    return_values['train_interval_len_var'] = interval_sizes.var().item()

    if consider_groups:
        y_upper = test_y_upper
        y_lower = test_y_lower
        coverage = (y_test < y_upper) & (y_lower < y_test)
        interval_lengths = y_upper - y_lower
        n_groups = int(x_test[:, group_feature].max().item()) + 1
        for group_number in range(n_groups):
            return_values['test_group_' + str(group_number) + '_coverage'] = (coverage[x_test[:, group_feature]
                                                                                       == group_number] == 1).sum().item() / len(
                coverage[x_test[:, group_feature] == group_number])
            return_values['test_group_' + str(group_number) + '_interval_len'] = interval_lengths[
                x_test[:, group_feature] == group_number].mean().item()
            return_values['test_group_' + str(group_number) + '_interval_len_var'] = interval_lengths[
                x_test[:, group_feature] == group_number].var().item()

        coverage = (y_train < train_y_upper) & (train_y_lower < y_train)
        interval_lengths = train_y_upper - train_y_lower
        n_groups = int(x_train[:, group_feature].max().item()) + 1
        for group_number in range(n_groups):
            return_values['train_group_' + str(group_number) + '_coverage'] = (coverage[x_train[:, group_feature]
                                                                                        == group_number] == 1).sum().item() / len(
                coverage[x_train[:, group_feature] == group_number])
            return_values['train_group_' + str(group_number) + '_interval_len'] = interval_lengths[
                x_train[:, group_feature] == group_number].mean().item()
            return_values['train_group_' + str(group_number) + '_interval_len_var'] = interval_lengths[
                x_train[:, group_feature] == group_number].var().item()

    return return_values
