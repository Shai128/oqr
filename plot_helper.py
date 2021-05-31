import six
import sys
import torch
import numpy as np

import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
import matplotlib.ticker as ticker

import matplotlib.pyplot as plt

import helper
import pandas as pd
import ast
import matplotlib as mpl
import sklearn.tree
import traceback
import math

from helper import create_folder_if_it_doesnt_exist, get_x_test
from utils.penalty_multipliers import real_corr_per_dataset_per_loss, syn_corr_per_dataset_per_loss


if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

np.warnings.filterwarnings('ignore')
sys.modules['sklearn.externals.six'] = six


class SquareRootScale(mscale.ScaleBase):
    """
    ScaleBase class for generating square root scale.
    """

    name = 'squareroot'

    def __init__(self, axis, **kwargs):
        # note in older versions of matplotlib (<3.1), this worked fine.
        # mscale.ScaleBase.__init__(self)

        # In newer versions (>=3.1), you also need to pass in `axis` as an arg
        mscale.ScaleBase.__init__(self, axis)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return max(0., vmin), vmax

    class SquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            return np.array(a) ** 0.5

        def inverted(self):
            return SquareRootScale.InvertedSquareRootTransform()

    class InvertedSquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform(self, a):
            return np.array(a) ** 2

        def inverted(self):
            return SquareRootScale.SquareRootTransform()

    def get_transform(self):
        return self.SquareRootTransform()

mscale.register_scale(SquareRootScale)


def plot_quantile_estimator_losses(quantile_estimator):
    pinball_loss = np.array(quantile_estimator.learner.pinball_loss_history)
    dependency_loss = np.zeros_like(pinball_loss)
    coverage_loss = np.zeros_like(pinball_loss)

    if len(quantile_estimator.learner.dependence_loss_history) > 0:
        dependency_loss_len = len(quantile_estimator.learner.dependence_loss_history)
        dependency_loss[-dependency_loss_len:] = np.array(quantile_estimator.learner.dependence_loss_history)
        coverage_loss[-dependency_loss_len:] = np.array(quantile_estimator.learner.coverage_loss_history)

    plt.semilogy(pinball_loss, label='pinball loss')
    plt.semilogy(dependency_loss, label='dependence loss')
    plt.semilogy(coverage_loss, label='coverage loss')
    plt.semilogy(quantile_estimator.learner.loss_history, label='total loss')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title("first loss on train")
    plt.legend(loc="upper right")
    plt.show()

    pinball_loss = np.array(quantile_estimator.learner.validation_pinball_losses)
    dependency_loss = np.zeros_like(pinball_loss)
    coverage_loss = np.zeros_like(pinball_loss)
    if len(quantile_estimator.learner.validation_dependency_losses) > 0:
        dependency_loss_len = len(quantile_estimator.learner.validation_dependency_losses)
        dependency_loss[-dependency_loss_len:] = np.array(quantile_estimator.learner.validation_dependency_losses)
        coverage_loss[-dependency_loss_len:] = np.array(quantile_estimator.learner.validation_coverage_losses)

    plt.semilogy(pinball_loss, label='pinball loss')
    plt.semilogy(dependency_loss, label='dependence loss')
    plt.semilogy(coverage_loss, label='coverage loss')

    plt.semilogy(quantile_estimator.learner.test_loss_history, label='total loss')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title("loss on validation")
    plt.legend(loc="upper right")
    plt.show()

    pinball_loss = np.array(quantile_estimator.learner.full_pinball_loss_history)
    dependency_loss = np.zeros_like(pinball_loss)
    coverage_loss = np.zeros_like(pinball_loss)
    dependency_loss_len = len(quantile_estimator.learner.full_dependence_loss_history)
    if dependency_loss_len > 0:
        dependency_loss[-dependency_loss_len:] = np.array(quantile_estimator.learner.full_dependence_loss_history)
        coverage_loss[-dependency_loss_len:] = np.array(quantile_estimator.learner.full_coverage_loss_history)

    plt.semilogy(pinball_loss, label='pinball loss')
    plt.semilogy(dependency_loss, label='dependence loss')
    plt.semilogy(coverage_loss, label='coverage loss')
    plt.semilogy(quantile_estimator.learner.full_loss_history, label='total loss')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title("final loss on train")
    plt.legend(loc="upper right")
    plt.show()


def plot_results_during_training(results_during_training, desired_coverage=0.9):
    for results_name in results_during_training:
        plt.plot(results_during_training[results_name])
        plt.title(results_name)
        plt.xlabel("epoch")
        if 'pearsons' in results_name:
            y_label = "Pearson's correlation"
            plt.axhline(y=0, color='b', linestyle='-')
        elif 'coverage' in results_name:
            y_label = 'coverage'
            plt.axhline(y=desired_coverage, color='r', linestyle='-')
            plt.axvline(len(results_during_training[results_name]) - 200, color='purple', linestyle='-')


        elif 'interval_lengths' in results_name:
            y_label = 'avg interval length'
            plt.axvline(len(results_during_training[results_name]) - 200, color='purple', linestyle='-')

        plt.ylabel(y_label)
        plt.show()


def plot_quantile_estimator_results_during_training(quantile_estimator, desired_coverage=0.9):
    learner = quantile_estimator.learner
    plot_results_during_training(learner.results_during_training, desired_coverage=desired_coverage)


def replace_None_by_0(a):
    if a is None:
        return 0
    if type(a) == str:
        a = ast.literal_eval(a.replace("tensor", "").replace("(", "").replace(")", ""))
    if type(a) == list and type(a[0]) == str:
        for i in range(len(a)):
            a[i] = ast.literal_eval(a[i].replace("tensor", "").replace("(", "").replace(")", ""))
    return a


features = ['coverage', 'interval_len']


def get_feature_df_to_boxplot(results, x_name='DS', features=features):
    df = pd.DataFrame(results)
    feature_df = {}
    for feature in features:

        feature_df[feature] = {}
        for i in range(len(df)):
            curr_row = df[feature].iloc[i]
            if type(df.index[i]) == float:
                row_name = df.index[i].round(3)
            else:
                row_name = df.index[i]
            if type(curr_row) == str:
                curr_row = ast.literal_eval(curr_row.replace("tensor", "").replace("(", "").replace(")", ""))
            if type(curr_row) == list and type(curr_row[0]) == str:
                for i in range(len(curr_row)):
                    if type(curr_row[i]) != str:
                        continue
                    curr_row[i] = ast.literal_eval(curr_row[i].replace("tensor", "").replace("(", "").replace(")", ""))

            feature_df[feature][x_name + str(row_name)] = {
                'Experiment ' + str(v): (float)(replace_None_by_0(curr_row[v])) for v, k in enumerate(curr_row)}
    return feature_df


def limit_y_plot(feature, significance=None):
    if feature == 'test_chi2_pvalue':
        plt.ylim(0, 1)
    elif feature == 'interval_len':
        plt.ylim(1.5, 2.5)
    elif 'coverage' in feature and significance is not None:
        mid = (1 - significance)
        range_around = significance / 2
        plt.ylim(mid - range_around, mid + range_around)
    elif 'corr' in feature:
        plt.ylim(-0.1, 0.1)



def plot_features(df, x_name='DS', x_label='DS#', limit_y=True, features=features, significance=None):
    df = get_feature_df_to_boxplot(df, x_name=x_name, features=features)
    for feature in features:

        pd.DataFrame(df[feature]).boxplot(figsize=(13.5, 3.5), return_type='axes')

        if limit_y:
            limit_y_plot(feature, significance)
        plt.ylabel(feature)
        plt.xlabel(x_label)
        plt.show()


def add_coverage_and_lengths_to_df(df):
    y, y_lower, y_upper = df['y_test'].item(), df['test_y_lower'].item(), df['test_y_upper'].item()
    y_test = np.array(list(map(float, y.strip('[]').split(','))))
    y_lower = np.array(list(map(float, y_lower.strip('[]').split(','))))
    y_upper = np.array(list(map(float, y_upper.strip('[]').split(','))))
    lengths = list(y_upper - y_lower)
    coverages = list(((y_test >= y_lower) & (y_test <= y_upper)).astype(np.float32))
    df['lengths'] = [lengths]
    df['coverages'] = [coverages]

    if 'test_hsic' not in df:
        n = 5000
        idx = np.random.permutation(len(lengths))[:n]
        n = len(idx)
        df['test_hsic'] = [helper.HSIC(torch.Tensor(lengths)[idx].reshape(n, 1).to(device), \
                                       torch.Tensor(coverages)[idx].reshape(n, 1).to(device)).item()]

    return df


def summarize_df(col_series):
    corr = np.abs(col_series['test_pearson_corr'])
    col_summary = {
        'Coverage': np.mean(col_series['coverage']),
        'Interval length average': np.mean(col_series['interval_len']),
        "Pearson's correlation average": np.mean(corr),
        "log(HSIC) average": np.mean(np.log10(col_series['test_hsic'])),
        'WSC average': np.mean(col_series['test_wsc']),
        'Delta WSC average': np.mean(col_series['test_wsc_diff']),
    }
    return col_summary



def calculate_node_coverage(seed, dataset_name, with_corr_coverages, no_corr_coverages,
                            length_diff, ninth_quantile, calibrated=False):
    x = get_x_test(dataset_name, seed)
    if calibrated:
        idx = np.arange(len(x))  # np.random.permutation(len(y_upper))
        n_half = int(np.floor(len(x) / 2))
        idx_test, idx_cal = idx[:n_half], idx[n_half:2 * n_half]
        x = x[idx_test]

    x = pd.DataFrame(x.numpy())

    large_diff_idx = length_diff[seed, :] > ninth_quantile[seed]
    large_diff_df = x[large_diff_idx]
    large_diff_df['had_diff'] = 1

    small_diff_df = x[~large_diff_idx]
    small_diff_df['had_diff'] = 0
    tree_df = pd.concat([large_diff_df, small_diff_df], axis=0)
    labels = tree_df['had_diff']
    tree_df = tree_df.drop(['had_diff'], axis=1)

    max_depth = 3
    tree = sklearn.tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)

    tree.fit(tree_df, labels)

    # finding the node with the best ratio (# increased length) / (#  non increased length)
    best_samples_ratio = 0
    best_node = None
    decision_path = tree.decision_path(x).toarray()
    n_nodes = decision_path.shape[1]

    for node in range(1, n_nodes):
        if decision_path[:, node].sum() < 0.05 * tree_df.shape[0]:
            continue

        n_increased_length_in_node = labels[decision_path[:, node] == 1].sum()
        n_not_increased_length_in_node = (decision_path[:, node] == 1).sum() - n_increased_length_in_node
        samples_ratio = n_increased_length_in_node / n_not_increased_length_in_node

        if best_node is None or samples_ratio > best_samples_ratio:
            best_node = node
            best_samples_ratio = samples_ratio
    if best_node is None:
        best_node = 0


    with_corr_coverage_in_node = with_corr_coverages[seed][decision_path[:, best_node] == 1].mean()
    no_corr_coverage_in_node =no_corr_coverages[seed][decision_path[:, best_node] == 1].mean()
    return with_corr_coverage_in_node, no_corr_coverage_in_node

def display_cov_vs_len_graphs(col,corr_1_cov_and_len, no_corr_cov_and_len,
                              desired_coverage, loss_method, dataset_name, examples_per_bin):
    fig1 = plt.figure()
    # fig2 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    # ax2 = fig2.add_subplot(1, 1, 1)
    axes = [ax1]

    for axe in axes:
        axe.set_xlabel("Length", fontsize=18)
        axe.set_ylabel("Coverage", fontsize=18)
        if 'blog' in dataset_name or 'facebook' in dataset_name or 'meps' in dataset_name:
            axe.set_xscale('squareroot')
        if 'meps' in dataset_name:
            plt.xticks(ticks=[1, 2, 4, 9, 16, 25])
    mpl.rc('font', **{'size': 18})

    t = torch.Tensor(no_corr_cov_and_len.sort_values(['interval_sizes'], axis=0)[
                         ['coverage', 'interval_sizes']].to_numpy()).split(examples_per_bin)
    l = torch.stack((list(map(lambda t: torch.stack((t[:, 0].mean(), t[:, 1].mean())).T, t))), dim=0)
    coverages, lengths = l.T.numpy()

    axes[0].axhline(y=desired_coverage, color='r', linestyle='--')
    axes[0].set_title("Coverage VS Interval Length")
    axes[0].plot(lengths, coverages, label="vanilla QR")

    t = torch.Tensor(corr_1_cov_and_len.sort_values(['interval_sizes'], axis=0)[
                         ['coverage', 'interval_sizes']].to_numpy()).split(examples_per_bin)
    l = torch.stack((list(map(lambda t: torch.stack((t[:, 0].mean(), t[:, 1].mean())).T, t))), dim=0)
    coverages, lengths = l.T.numpy()

    axes[0].axhline(y=desired_coverage, color='r', linestyle='--')
    axes[0].set_title("Coverage VS Interval Length")
    axes[0].plot(lengths, coverages, label="orthogonal QR")
    axes[0].legend(loc="lower right")
    save_dir = f'./results/figures/real/final_results/{loss_method}/Coverage VS Interval Length/{dataset_name}'
    create_folder_if_it_doesnt_exist(save_dir)
    fig1.set_size_inches(6.4, 4.8)
    fig1.savefig(f'{save_dir}/{col}.png', dpi=300)

    plt.show()



def display_results_over_datasets(dataset_names, loss_method='qr',
                                  corr_mults=None, hsic_mults=None, seeds=range(0, 30),
                                  desired_coverage=0.9, calibrated=False, is_synthetic=False, print_latex=False):

    if is_synthetic:
        display_results_func = display_syn_results_over_dataset
    else:
        display_results_func = display_results_over_dataset

    assert loss_method == 'qr' or loss_method == 'int'
    std_errors = {}
    for dataset_name in dataset_names:

        res = display_results_func(dataset_name, loss_method, True, corr_mults, hsic_mults, seeds,
                                     desired_coverage, calibrated)
        if len(res) == 3:
            std_errors[dataset_name], baseline_method, improved_method = res
    if len(std_errors) == 0:
        return
    metrics = ['test_pearson_corr', 'test_hsic', 'test_wsc_diff']
    if not is_synthetic:
        metrics += ['Delta_ILS_coverage', 'Delta_node_coverage']

    std_errors_df = {}

    metrics = ['coverage', 'length'] + metrics

    def print_method_res(dataset_name, metric, method_number):
        mean1 = std_errors[dataset_name]['mean'][metric][method_number]
        mean2 = std_errors[dataset_name]['mean'][metric][1-method_number]
        if mean1 < mean2 and print_latex:
            return "\\textbf{"+float_to_str(mean1)+ f" ({float_to_str(std_errors[dataset_name][metric][method_number])})"+"}"
        else:
            return float_to_str(mean1) + f" ({float_to_str(std_errors[dataset_name][metric][method_number])})"

    for metric in metrics:

        std_errors_df[metric] = {'QR':[print_method_res(dataset_name, metric, 0)
                                    for dataset_name in dataset_names],

                                  'OQR': [print_method_res(dataset_name, metric, 1)
                                  for dataset_name in dataset_names]
                                 }

    map_metric_name = {
                    'coverage': 'Coverage',
                    'length': 'Length',
                    'test_hsic': 'HSIC',
                     'test_wsc_diff': 'Delta WSC',
                     'test_pearson_corr': "Pearson's corr",
                     'Delta_ILS_coverage': 'Delta ILS-Coverage',
                     'Delta_node_coverage': 'Delta Node-Coverage'}
    if print_latex:
        metrics.remove('coverage')
        metrics.remove('length')
    if print_latex:
        bolded_dataset_names = ["\\textbf{"+dataset_name+"}" for dataset_name in dataset_names]
    else:
        bolded_dataset_names = dataset_names
    std_errors_df = pd.concat(
        dict(zip([map_metric_name[metric] for metric in metrics],
                 [pd.DataFrame(std_errors_df[metric], index=bolded_dataset_names) for metric in metrics])),axis=1)
    if print_latex:
        print(std_errors_df.to_latex(index=True).replace("\\textbackslash ", "\\").replace("\{", "{").replace("\}", "}"))

    print("Mean values and standard errors:")
    display(std_errors_df)


def float_to_str(n):
    if abs(n) < 1e-3 and abs(n) > 1e-20 and not math.isnan(n):
        if n > 0:
            sign = ''
        else:
            sign = '-'
        return sign + '1e' + str(int(np.log10(abs(n))))
    else:

        n_str = "{:.3f}".format(n)
        if abs(n) < 1:
            n_str = n_str.replace("0.", ".")
        return n_str
def display_results_over_dataset(dataset_name, loss_method='qr', display_tables_only=False,
                                 corr_mults=None, hsic_mults=None, seeds=range(0, 30),
                                 desired_coverage=0.9, calibrated=False):
    features = ['coverage', 'interval_len', 'coverages', 'lengths',
                'test_hsic', 'test_wsc', 'test_wsc_diff',
                'test_pearson_corr', 'test_pearson_pvalue'
                ]
    df = {}
    def add_to_def(loss_name, corr_mult, hsic_mult, dataset_name, display_name=''):
        folder_name = get_folder_name_from_args(loss_name, corr_mult, hsic_mult, calibrated)
        try:
            results_path = helper.results_path + 'real_data/' + \
                           dataset_name + '/' + folder_name + '/'

            n = ''

            if 'meps' in dataset_name:
                n = dataset_name[-2:]

            if 'facebook' in dataset_name:
                n = dataset_name[-1:]

            if 'meps' in dataset_name:
                display_dataset_name = dataset_name[0:3]
            else:
                display_dataset_name = dataset_name[0:4]

            row_name = display_name + '_' + display_dataset_name + n
            df[row_name] = {}
            for seed in seeds:
                try:
                    path = results_path + f'seed={seed}.csv'
                    res = pd.read_csv(path)

                    need_to_resave = 'test_hsic' not in res

                    cov = float(res['coverage'].item().replace("tensor(", "").replace(")", ""))
                    res['test_wsc_diff'] = abs(res['test_wsc'] - cov)
                    res = add_coverage_and_lengths_to_df(res)
                    res = res[[col for col in res.columns if 'Unnamed' not in col]]
                    if need_to_resave:
                        res.to_csv(path)

                except Exception as e:
                    is_necessary_mult = corr_mult ==\
                                         real_corr_per_dataset_per_loss[loss_name.replace("batch_","")][dataset_name] or\
                                        (corr_mult == 0. and hsic_mult==0.) or\
                                        hsic_mult == real_corr_per_dataset_per_loss['hsic_qr'][dataset_name]

                    if is_necessary_mult:
                        print(f"Didn't find results in path: {path}.")
                    # traceback.print_exc()
                    break

                for feature in features:

                    if feature not in df[row_name]:
                        df[row_name][feature] = []
                    df[row_name][feature] += [res.iloc[0][feature]]

            if len(df[row_name]) == 0:
                del df[row_name]

        except Exception as e:
            print(e)
            del df[row_name]


    method_name = loss_method

    loss_name = f'batch_{method_name}'

    if corr_mults is None:
        corr_mults = [0.0, 0.01, 0.1, 0.5, 1., 2., 3.]
    for corr_mult in corr_mults:
        add_to_def(loss_name, corr_mult, 0., dataset_name, display_name=method_name + f'+corr{(corr_mult)}', )

    if hsic_mults is None:
        hsic_mults = [0.0, 0.01, 0.1, 0.5, 1., 2., 3.]
    for hsic_mult in hsic_mults:
        add_to_def(loss_name, 0., hsic_mult, dataset_name, display_name=method_name + f'+hsic{(hsic_mult)}')

    df = pd.DataFrame(df)

    cols = pd.DataFrame(df).columns.to_list()
    cols.sort(key=lambda x: x[-4:])

    df = df[cols]

    df_to_plot = pd.DataFrame(df).T.drop(['coverages', 'lengths'], axis=1, errors='ignore')

    if display_tables_only:
        backend_ = mpl.get_backend()
        mpl.use("Agg")
    plot_features(df_to_plot, x_name='', x_label='', limit_y=False, features=df_to_plot.columns)
    if display_tables_only:
        mpl.use(backend_)

    if 'coverages' not in pd.DataFrame(df).T.columns:
        return
    no_corr_column = pd.DataFrame(df).columns[0]
    no_corr_coverages = np.array(pd.DataFrame(df)[no_corr_column].loc['coverages'])
    no_corr_lengths = np.array(pd.DataFrame(df)[no_corr_column].loc['lengths'])

    examples_per_bin = int((1 / 100) * no_corr_coverages.shape[0] * no_corr_coverages.shape[1])
    no_corr_cov_and_len = pd.DataFrame({'coverage': no_corr_coverages.flatten(),
                                        'interval_sizes': no_corr_lengths.flatten()})

    minorities_info = {}
    summary_df = {}
    summary_df[no_corr_column] = summarize_df(df[no_corr_column])

    for col in pd.DataFrame(df).columns[1:]:
        print(col)
        minorities_info[col] = {}
        with_corr_coverages = np.array(pd.DataFrame(df)[col].loc['coverages']).astype(np.float32)
        with_corr_lengths = np.array(pd.DataFrame(df)[col].loc['lengths']).astype(np.float32)
        corr_1_cov_and_len = pd.DataFrame({'coverage': with_corr_coverages.flatten(),
                                           'interval_sizes': with_corr_lengths.flatten()})

        display_cov_vs_len_graphs(col, corr_1_cov_and_len, no_corr_cov_and_len,
                              desired_coverage, loss_method, dataset_name, examples_per_bin)

        try:
            length_diff = with_corr_lengths - no_corr_lengths

            ninth_quantile = np.sort(length_diff, axis=1)[:, int((with_corr_coverages.shape[1]) * 0.9)]

            with_corr_minority_coverage = []
            no_corr_minority_coverage = []
            for seed in range(ninth_quantile.shape[0]):
                with_corr_minority_coverage += [
                    with_corr_coverages[seed, :][(length_diff[seed, :].T > ninth_quantile[seed]).T].mean() * 100]
                no_corr_minority_coverage += [
                    no_corr_coverages[seed, :][(length_diff[seed, :].T > ninth_quantile[seed]).T].mean() * 100]

            with_corr_coverage_in_node = []
            no_corr_coverage_in_node = []
            for seed in range(ninth_quantile.shape[0]):
                with_corr_node_cov, no_corr_node_cov =\
                    calculate_node_coverage(seed, dataset_name, with_corr_coverages, no_corr_coverages,
                                            length_diff, ninth_quantile, calibrated)
                with_corr_coverage_in_node += [with_corr_node_cov]
                no_corr_coverage_in_node += [no_corr_node_cov]

            with_corr_delta_node_cov = np.mean(abs(np.array(with_corr_coverage_in_node) - with_corr_coverages.mean(axis=1)))
            no_corr_delta_node_cov = np.mean(abs(np.array(no_corr_coverage_in_node) - no_corr_coverages.mean(axis=1)))

            minorities_info[col]['Delta_node_coverage_ratio'] = with_corr_delta_node_cov/no_corr_delta_node_cov
            minorities_info[col]['Delta_node_coverages_improved'] = abs(np.array(with_corr_coverage_in_node) -\
                                                                        with_corr_coverages.mean(axis=1))
            minorities_info[col]['Delta_node_coverages_baseline'] = abs(np.array(no_corr_coverage_in_node) -\
                                                                        no_corr_coverages.mean(axis=1))

            with_corr_delta_ILS_coverage = []
            no_corr_delta_ILS_coverage = []
            for seed in range(with_corr_coverages.shape[0]):
                with_corr_delta_ILS_coverage.append(
                    abs(with_corr_coverages[seed, :][(length_diff[seed, :].T > ninth_quantile[seed]).T].mean() - \
                        with_corr_coverages[seed, :].mean()))

                no_corr_delta_ILS_coverage.append(
                    abs(no_corr_coverages[seed, :][(length_diff[seed, :].T > ninth_quantile[seed]).T].mean() - \
                        no_corr_coverages[seed, :].mean()))

            # (orthogonal QR Delta ILS-Coverage) / (vanilla QR Delta ILS-coverage)
            minorities_info[col]['Delta_ILS_coverage_ratio'] = \
                np.mean(with_corr_delta_ILS_coverage) / \
                np.mean(no_corr_delta_ILS_coverage)

            minorities_info[col]['Delta_ILS_coverages_improved'] = np.array(with_corr_delta_ILS_coverage)
            minorities_info[col]['Delta_ILS_coverages_baseline'] = np.array(no_corr_delta_ILS_coverage)

            summary_df[col] = summarize_df(df[col])

            if len(pd.DataFrame(df).columns) == 2:  # if we compare vanilla vs orthogonal with the best corr mult
                summary_df[col]["Delta ILS-Coverage"] = np.mean(with_corr_delta_ILS_coverage)
                summary_df[no_corr_column]["Delta ILS-Coverage"] = np.mean(no_corr_delta_ILS_coverage)
                summary_df[col]["Delta Node-Coverage"] = with_corr_delta_node_cov
                summary_df[no_corr_column]["Delta Node-Coverage"] = no_corr_delta_node_cov


        except Exception:
            traceback.print_exc()
            print(f"There are not enough seed results for {col}.")
            continue

    if len(summary_df) != len(pd.DataFrame(df).columns):
        print(f"No results for dataset {dataset_name}.")
        return
    for col in pd.DataFrame(df).columns:

        for key in summary_df[col]:
            summary_df[col][key] = float_to_str(summary_df[col][key])
            # if abs(summary_df[col][key]) < 1e-3 and abs(summary_df[col][key]) > 1e-20 and not math.isnan(
            #         summary_df[col][key]):
            #     if summary_df[col][key] > 0:
            #         sign = ''
            #     else:
            #         sign = '-'
            #     summary_df[col][key] = sign + '1e' + str(int(np.log10(abs(summary_df[col][key]))))
            # else:
            #     summary_df[col][key] = "{:.3f}".format(summary_df[col][key])

    if len(pd.DataFrame(summary_df)) > 0:
        display(pd.DataFrame(summary_df))

    corr_per_dataset_per_loss = real_corr_per_dataset_per_loss
    try:
        measurements = ['test_hsic', 'test_wsc_diff', 'test_pearson_corr']
        measurements_that_should_decrease = measurements

        # if there is an hsic column
        compute_hsic = len([col for col in pd.DataFrame(df).columns if 'hsic' in col]) > 0
        baseline_method = 'Vanilla QR'

        improved_method = 'Orthogonal QR (corr)'
        if compute_hsic:

            hsic_mult = corr_per_dataset_per_loss['hsic_qr'][dataset_name]
            corr_mult = corr_per_dataset_per_loss['qr'][dataset_name]

            if len([col for col in pd.DataFrame(df).columns if str(corr_mult) in col and 'corr' in col]) == 0:
                baseline_method_column_name = pd.DataFrame(df).columns[0]
                improved_method_column_name = \
                [col for col in pd.DataFrame(df).columns if str(hsic_mult) in col and 'hsic' in col][0]
                baseline_method = 'Vanilla QR'
                improved_method = 'Orthogonal QR (HSIC)'
            else:
                improved_method_column_name = \
                [col for col in pd.DataFrame(df).columns if str(corr_mult) in col and 'corr' in col][0]
                baseline_method_column_name = \
                [col for col in pd.DataFrame(df).columns if str(hsic_mult) in col and 'hsic' in col][0]
                baseline_method = 'Orthogonal QR (HSIC)'
                improved_method = 'Orthogonal QR (corr)'

            baseline_method_column = pd.DataFrame(df)[baseline_method_column_name]
            improved_method_column = pd.DataFrame(df)[improved_method_column_name]

        else:
            corr_per_dataset = corr_per_dataset_per_loss[loss_method]
            corr = corr_per_dataset[dataset_name]
            baseline_method_column = pd.DataFrame(df)[pd.DataFrame(df).columns[0]]

            improved_method_column_name = [col for col in pd.DataFrame(df).columns if str(corr) in col][0]
            improved_method_column = pd.DataFrame(df)[improved_method_column_name]

        final_dataset_df = {}

        for measurement in measurements_that_should_decrease:
            increasement_ratio = np.mean(np.abs(improved_method_column[measurement])) / \
                                 np.mean(np.abs(baseline_method_column[measurement]))
            # multiply by negative 1 because smaller metric -> better conditional coverage
            increasement_percentage = -(increasement_ratio - 1) * 100
            curr_key_name = measurement
            final_dataset_df[curr_key_name] = '{:2.2f}'.format(np.round(increasement_percentage, 2))
            if increasement_percentage > 0:
                final_dataset_df[curr_key_name] = '+' + final_dataset_df[curr_key_name]

        add_delta_ILS_and_node = 'Vanilla' in baseline_method
        if add_delta_ILS_and_node:
            percentage = -(minorities_info[improved_method_column_name]['Delta_node_coverage_ratio'] - 1) * 100
            curr_key_name = 'Node coverage diff'
            final_dataset_df[curr_key_name] = '{:2.2f}'.format(np.round(percentage, 2))
            if percentage > 0:
                final_dataset_df[curr_key_name] = '+' + final_dataset_df[curr_key_name]

            percentage = -(minorities_info[improved_method_column_name]['Delta_ILS_coverage_ratio'] - 1) * 100
            curr_key_name = 'Increased length coverage diff'
            final_dataset_df[curr_key_name] = '{:2.2f}'.format(np.round(percentage, 2))
            if percentage > 0:
                final_dataset_df[curr_key_name] = '+' + final_dataset_df[curr_key_name]

        parameters_order = ["Pearson's corr",
                            'HSIC',
                            'Delta WSC',
                            'Delta ILS-Coverage',
                            'Delta Node-Coverage',
                            ]

        if not add_delta_ILS_and_node:
            parameters_order.remove('Delta Node-Coverage')
            parameters_order.remove('Delta ILS-Coverage')

        if calibrated:
            baseline_method = baseline_method.replace("QR", "CQR")
            improved_method = improved_method.replace("QR", "CQR")

        baseline_description = baseline_method
        improved_description = improved_method
        std_errors = calc_std_errors(baseline_method, improved_method, baseline_method_column, improved_method_column,
                                     improved_method_column_name, minorities_info)
        final_dataset_df = pd.concat({
            'Coverage': pd.DataFrame(
                {baseline_description: ['{:2.2f}%'.format(np.mean(baseline_method_column.coverage) * 100)],
                 improved_description: ['{:2.2f}%'.format(np.mean(improved_method_column.coverage) * 100)]},
                index=[dataset_name]),
            'Length': pd.DataFrame({baseline_description: ['{:2.2f}'.format(np.mean(baseline_method_column.interval_len))],
                                    improved_description: ['{:2.2f}'.format(np.mean(improved_method_column.interval_len))]},
                                   index=[dataset_name]),
            '% Improvement': pd.DataFrame(final_dataset_df, index=[dataset_name]).rename(
                columns={'test_hsic': 'HSIC',
                         'test_wsc_diff': 'Delta WSC',
                         'test_pearson_corr': "Pearson's corr",
                         'Increased length coverage diff': 'Delta ILS-Coverage',
                         'Node coverage diff': 'Delta Node-Coverage'})[parameters_order]


        }, axis=1)
        if calibrated:
            loss_method = 'cal_'+loss_method
        save_dir = f"results/final_results/{dataset_name}"
        create_folder_if_it_doesnt_exist(save_dir)
        final_dataset_df.to_csv(
            f'{save_dir}/{loss_method} {baseline_method} vs {improved_method} final_df.csv')
        return std_errors, baseline_method, improved_method
    except Exception:
        traceback.print_exc()
        print("Don't have the full results to construct the final comparison table.")
    # display(pd.DataFrame(final_dataset_df, index=[dataset_names[0]]))
    return {}

def ratio_stderr(X, Y):
    X = np.array(X)
    Y = np.array(Y)

    var = (((X.mean() / Y.mean()))**2)*(X.var()/(X.mean()**2)+Y.var()/(Y.mean()**2) - 2*np.cov(X,Y)[0][1]/(X.mean()*Y.mean()))
    return np.sqrt(var) / np.sqrt(len(X))

def calc_std_errors(baseline_method_name, improved_method_name, baseline_method_col, improved_method_col,
                    improved_method_column_name=None, minorities_info=None):
    sqrt_n = np.sqrt(len(baseline_method_col.coverage))
    assert baseline_method_col.coverage[0] < 1 and improved_method_col.coverage[0] < 1

    metrics = ['test_hsic', 'test_wsc_diff', 'test_pearson_corr']
    std_errors = {}
    # if not separate_SE:
    #     std_errors['coverage'] = {baseline_method_name: np.std(baseline_method_col.coverage * 100) / sqrt_n,
    #                      improved_method_name: np.std(improved_method_col.coverage * 100) / sqrt_n}
    #
    #     std_errors['length'] = {baseline_method_name: np.std(baseline_method_col.interval_len) / sqrt_n,
    #                    improved_method_name: np.std(improved_method_col.interval_len) / sqrt_n}

    metrics = ['coverage', 'length'] + metrics

    baseline_method_col = baseline_method_col.copy()
    improved_method_col = improved_method_col.copy()
    baseline_method_col['coverage'] = np.array(baseline_method_col.coverage) * 100
    improved_method_col['coverage'] = np.array(improved_method_col.coverage) * 100

    baseline_method_col['length'] = baseline_method_col.interval_len
    improved_method_col['length'] = improved_method_col.interval_len


    if improved_method_column_name is not None and minorities_info is not None:
        col = improved_method_column_name

        baseline_method_col['Delta_ILS_coverage'] = minorities_info[col]['Delta_ILS_coverages_baseline']
        improved_method_col['Delta_ILS_coverage'] = minorities_info[col]['Delta_ILS_coverages_improved']

        baseline_method_col['Delta_node_coverage'] = minorities_info[col]['Delta_node_coverages_baseline']
        improved_method_col['Delta_node_coverage'] = minorities_info[col]['Delta_node_coverages_improved']

        metrics += ['Delta_ILS_coverage', 'Delta_node_coverage']
    std_errors['mean'] = {}

    mult_by_100_metrics = ['test_wsc_diff', 'Delta_ILS_coverage', 'Delta_node_coverage']
    for metric in [metric for metric in mult_by_100_metrics if metric in metrics]:
        assert np.max(np.abs(baseline_method_col[metric])) <= 1
        assert np.max(np.abs(improved_method_col[metric])) <= 1
        baseline_method_col[metric] = np.array(baseline_method_col[metric]) * 100
        improved_method_col[metric] = np.array(improved_method_col[metric]) * 100

    for metric in metrics:
        std_errors[metric] = [np.std(baseline_method_col[metric]) / sqrt_n,
                              np.std(improved_method_col[metric]) / sqrt_n]
        std_errors['mean'][metric] = [np.mean(np.abs(baseline_method_col[metric])),
                              np.mean(np.abs(improved_method_col[metric]))]
        # print(f"baseline {metric} std err: {np.std(baseline_method_col[metric]) / sqrt_n}")
        # print(f"improved {metric} std err: {np.std(improved_method_col[metric]) / sqrt_n}")




    return std_errors


def map_sign_to_color(sign):
    if sign[0] not in ['+', '-']:
        color = 'black'
    elif sign[0] == '+':
        color = 'green'
    else:
        color = 'red'
    return 'color: %s' % color


def display_final_results(loss_method, baseline_method, improved_method):
    dataset_names = ['facebook_1', 'facebook_2', 'blog_data', 'bio',
                     'kin8nm', 'naval', 'meps_19', 'meps_20', 'meps_21']
    final_df = pd.DataFrame()

    for dataset_name in dataset_names:
        try:
            final_df = pd.concat([final_df, pd.read_csv(
                f'results/final_results/{dataset_name}/{loss_method} {baseline_method} vs {improved_method} final_df.csv',
                header=[0, 1], dtype=str)])
        except Exception:
            pass
    try:
        final_df = final_df.rename(columns={'Unnamed: 0_level_0': 'dataset name'}).set_index('dataset name')
        indexes = [name[0] for name in list(final_df.index)]
        final_df = final_df.set_index(pd.Index(indexes))
        styled_final_df = final_df.style.applymap(map_sign_to_color)
        styled_final_df = styled_final_df.set_table_styles([
            {'selector': 'th',
             'props': [
                 ('text-align', 'center'),
                 ('border-width', '0px'),
             ]
             }]
        )
    except Exception:
        print("No results to display.")
        return
    display(styled_final_df)
    styled_final_df.to_excel(f'results/final_results/{loss_method}, {baseline_method}_vs_{improved_method} styled_final_df.xlsx')

    return styled_final_df


def get_folder_name_from_args(loss, corr_mult, hsic_mult, calibrated=False):
    corr_mult = float(corr_mult)
    hsic_mult = float(hsic_mult)
    folder_name = f'loss={loss}_bs=1024_corr_mult={corr_mult}_hsic_mult={hsic_mult}'
    if calibrated:
        folder_name = 'cal_'+folder_name
    return folder_name



def display_syn_results_over_datasets(dataset_names, loss_method='qr',
                                      corr_multipliers=None, hsic_multipliers=None):
    display_results_over_datasets(dataset_names, loss_method,
                                      corr_multipliers, hsic_multipliers,
                                  is_synthetic=True)
    # assert loss_method == 'qr' or loss_method == 'int'
    # std_errors = {}
    # for dataset_name in dataset_names:
    #     std_errors[dataset_name] = display_syn_results_over_dataset(dataset_name, loss_method, True,
    #                                      corr_multipliers, hsic_multipliers)
    #
    # max_coverage_std_error = max([error for error in std_errors['dataset_name']['coverage'].values()])
    # max_length_std_error = max([error for error in std_errors['dataset_name']['length'].values()])
    # print(f"max coverage standard error: {max_coverage_std_error}")
    # print(f"max length standard error: {max_length_std_error}")



def display_syn_results_over_dataset(dataset_name, loss_method='qr', display_tables_only=False,
                                     corr_multipliers=None, hsic_multipliers=None, seeds=range(30),
                                     desired_coverage=0.9, calibrated=False):

    features = ['coverage', 'interval_len', 'coverages', 'lengths',
                'test_hsic', 'test_wsc', 'test_wsc_diff',
                'test_pearson_corr', 'test_pearson_pvalue'
                ]

    df = {}
    # runs_per_set = 30
    # seeds = range(0, runs_per_set)
    n_groups = 2
    for group_number in range(n_groups):
        features += ['test_group_' + str(group_number) + '_coverage']
        features += ['test_group_' + str(group_number) + '_interval_len']

    def add_to_def(folder_name, dataset_name, display_name=''):

        try:
            results_path = helper.results_path + \
                           'syn_data/minority_group_uncertainty=' + str(dataset_name) + '/' + folder_name + '/'

            display_dataset_name = dataset_name

            row_name = display_name + display_dataset_name
            df[row_name] = {}
            for seed in seeds:
                try:
                    path = results_path + f'seed={seed}.csv'
                    res = pd.read_csv(path)
                    cov = float(res['coverage'].item().replace("tensor(", "").replace(")", ""))
                    res['test_wsc_diff'] = abs(float(res['test_wsc']) - cov)
                    res = add_coverage_and_lengths_to_df(res)

                except Exception as e:
                    print(f"Didn't find results in path: {path}.")
                    break
                res = res.drop(['Unnamed: 0'], axis=1)

                for feature in features:

                    if feature not in df[row_name]:
                        df[row_name][feature] = []
                    df[row_name][feature] += [res.iloc[0][feature]]

            if len(df[row_name]) == 0:
                del df[row_name]

        except Exception as e:
            print(e)
            del df[row_name]

    method_name = loss_method

    loss_name = f'batch_{method_name}'

    if corr_multipliers is None:
        corr_multipliers = [0.0, 0.01, 0.1, 0.5, 1., 2., 3.]

    for corr_multiplier in corr_multipliers:
        folder_name = get_folder_name_from_args(loss_name, corr_multiplier, 0.0)
        add_to_def(folder_name, dataset_name,
                   display_name=method_name + f'+corr{(corr_multiplier)}_λ=')

    if hsic_multipliers is None:
        hsic_multipliers = [0.0, 0.01, 0.1, 0.5, 1., 2.]

    for hsic_multiplier in hsic_multipliers:
        folder_name =  get_folder_name_from_args(loss_name, 0.0, hsic_multipliers)
        add_to_def(folder_name, dataset_name,
                   display_name=method_name + f'+hsic{(hsic_multiplier)}_λ=')

    df = pd.DataFrame(df)

    df_to_plot = pd.DataFrame(df).T.drop(['coverages', 'lengths'], axis=1, errors='ignore')

    if display_tables_only:
        backend_ = mpl.get_backend()
        mpl.use("Agg")
    plot_features(df_to_plot, x_name='', x_label='', limit_y=False, features=df_to_plot.columns)
    if display_tables_only:
        mpl.use(backend_)

    summary_df = {}

    for col in pd.DataFrame(df).columns:

        summary_df[col] = summarize_df(df[col])

        for i in range(2):
            summary_df[col].update({
                f'Group {i} coverage average': np.mean(df[col][f'test_group_{i}_coverage']),
                f'Group {i} interval length average': np.mean(df[col][f'test_group_{i}_interval_len']),

            })

        for key in summary_df[col]:
            if abs(summary_df[col][key]) < 1e-3 and abs(summary_df[col][key]) > 1e-20 and not math.isnan(
                    summary_df[col][key]):
                if summary_df[col][key] > 0:
                    sign = ''
                else:
                    sign = '-'
                summary_df[col][key] = sign + '1e' + str(int(np.log10(abs(summary_df[col][key]))))
            else:
                summary_df[col][key] = "{:.3f}".format(summary_df[col][key])
    if len(pd.DataFrame(summary_df)) > 0:
        display(pd.DataFrame(summary_df))

    if 'coverages' not in pd.DataFrame(df).T.columns:
        return

    corr_per_dataset_per_loss = syn_corr_per_dataset_per_loss
    try:

        measurements = ['test_hsic', 'test_wsc_diff', 'test_pearson_corr']
        measurements_that_should_decrease = measurements

        # if there is an hsic column
        compute_hsic = len([col for col in pd.DataFrame(df).columns if 'hsic' in col]) > 0
        baseline_method = 'Vanilla QR'
        improved_method = 'Orthogonal QR (corr)'
        if compute_hsic:
            improved_method = 'Orthogonal QR (HSIC)'
            hsic_mult = corr_per_dataset_per_loss['hsic_qr'][dataset_name]
            corr_mult = corr_per_dataset_per_loss['qr'][dataset_name]

            if len([col for col in pd.DataFrame(df).columns if str(corr_mult) in col and 'corr' in col]) == 0:

                baseline_method_column_name = pd.DataFrame(df).columns[0]
            else:
                baseline_method = 'corr'
                baseline_method_column_name = \
                [col for col in pd.DataFrame(df).columns if str(corr_mult) in col and 'corr' in col][0]

            baseline_method_column = pd.DataFrame(df)[baseline_method_column_name]

            improved_method_column_name = \
            [col for col in pd.DataFrame(df).columns if str(hsic_mult) in col and 'hsic' in col][0]
            improved_method_column = pd.DataFrame(df)[improved_method_column_name]

        else:
            corr_per_dataset = corr_per_dataset_per_loss[loss_method]
            corr = corr_per_dataset[dataset_name]
            baseline_method_column = pd.DataFrame(df)[pd.DataFrame(df).columns[0]]

            improved_method_column_name = [col for col in pd.DataFrame(df).columns if str(corr) in col][0]
            improved_method_column = pd.DataFrame(df)[improved_method_column_name]

        final_dataset_df = {}

        for measurement in measurements_that_should_decrease:
            increasement_ratio = np.mean(np.abs(improved_method_column[measurement])) / \
                                 np.mean(np.abs(baseline_method_column[measurement]))
            increasement_percentage = -(increasement_ratio - 1) * 100

            curr_key_name = measurement
            final_dataset_df[curr_key_name] = '{:2.2f}'.format(np.round(increasement_percentage, 2))
            if increasement_percentage > 0:
                final_dataset_df[curr_key_name] = '+' + final_dataset_df[curr_key_name]

        parameters_order = ["Pearson's corr",
                            'HSIC',
                            'Delta WSC']
        std_errors = calc_std_errors(baseline_method, improved_method,
                                     baseline_method_column, improved_method_column)

        row_name = 'λ=' + dataset_name
        final_dataset_df = pd.concat({
            'Majority Coverage': pd.DataFrame({baseline_method: ['{:2.2f}%'.format(
                np.mean(baseline_method_column['test_group_0_coverage']) * 100)],
                improved_method: ['{:2.2f}%'.format(
                    np.mean(improved_method_column['test_group_0_coverage']) * 100)]},
                index=[row_name]),
            'Minority Coverage': pd.DataFrame({baseline_method: ['{:2.2f}%'.format(
                np.mean(baseline_method_column['test_group_1_coverage']) * 100)],
                improved_method: ['{:2.2f}%'.format(
                    np.mean(improved_method_column['test_group_1_coverage']) * 100)]},
                index=[row_name]),

            'Majority Lengths': pd.DataFrame({baseline_method: ['{:2.2f}'.format(
                np.mean(baseline_method_column['test_group_0_interval_len']))],
                improved_method: ['{:2.2f}'.format(
                    np.mean(improved_method_column['test_group_0_interval_len']))]},
                index=[row_name]),
            'Minority Lengths': pd.DataFrame({baseline_method: ['{:2.2f}'.format(
                np.mean(baseline_method_column['test_group_1_interval_len']))],
                improved_method: ['{:2.2f}'.format(
                    np.mean(improved_method_column['test_group_1_interval_len']))]},
                index=[row_name]),

            '% Improvement': pd.DataFrame(final_dataset_df, index=[row_name]).rename(
                columns={'test_hsic': 'HSIC',
                         'test_wsc_diff': 'Delta WSC',
                         'test_pearson_corr': "Pearson's corr"})[parameters_order]

        }, axis=1)

        save_dir = f"results/final_results/{dataset_name}"
        create_folder_if_it_doesnt_exist(save_dir)
        final_dataset_df.to_csv(
            f'{save_dir}/{loss_method} {baseline_method} vs {improved_method} final_df.csv')
        return std_errors, 'QR', 'OQR'

    except Exception:
        print("Don't have the full results to construct the final comparison table.")
    # display(pd.DataFrame(final_dataset_df, index=[row_name]))
    return {}
def map_sign_to_color(sign):
    if sign[0] not in ['+', '-']:
        color = 'black'
    elif sign[0] == '+':
        color = 'green'
    else:
        color = 'red'
    return 'color: %s' % color


def synthetic_display_final_results(loss_method, baseline_method, improved_method):
    dataset_names = ['3', '10']

    final_df = pd.DataFrame()

    for dataset_name in dataset_names:
        try:
            final_df = pd.concat([final_df, pd.read_csv(
                f'results/final_results/{dataset_name}/{loss_method} {baseline_method} vs {improved_method} final_df.csv',
                header=[0, 1], dtype=str)])
        except Exception:
            pass
    try:
        final_df = final_df.rename(columns={'Unnamed: 0_level_0': 'dataset name'}).set_index('dataset name')
        indexes = ["$" + name[0].replace("λ", "\\lambda") + "$" for name in list(final_df.index)]
        final_df = final_df.set_index(pd.Index(indexes))
        styled_final_df = final_df.style.applymap(map_sign_to_color)
        styled_final_df = styled_final_df.set_table_styles([
            {'selector': 'th',
             'props': [
                 ('text-align', 'center'),
                 ('border-width', '0px'),
             ]
             }]
        )
    except Exception:
        print("No results to display.")
        return None
    display(styled_final_df)

    styled_final_df.to_excel(f'results/final_results/syn {loss_method}, {baseline_method}_vs_{improved_method} styled_final_df.xlsx')

    return styled_final_df
