from copy import deepcopy

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

from cond_coverage_metrics import calculate_node_coverage
from helper import create_folder_if_it_doesnt_exist, get_x_test
from utils.penalty_multipliers import real_corr_per_dataset_per_loss, syn_corr_per_dataset_per_loss, \
    real_hsic_per_dataset_per_loss, syn_hsic_per_dataset_per_loss

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

np.warnings.filterwarnings('ignore')
sys.modules['sklearn.externals.six'] = six

VANILLA_QR = 'Vanilla QR'
OQR_CORR = 'OQR (corr)'
OQR_HSIC = 'OQR (HSIC)'

metrics_rename_map = {
    'coverage': 'Coverage',
    'interval_len': 'Length',
    'length': 'Length',
    'test_hsic': 'HSIC',
    'test_wsc_diff': 'ΔWSC',
    'test_pearson_corr': "Pearson's corr",
    'Delta_ILS_coverage_ratio': 'ΔILS-Coverage',
    'Delta_node_coverage_ratio': 'ΔNode-Coverage',
    'Delta_ILS_coverage': 'ΔILS-Coverage',
    'Delta_node_coverage': 'ΔNode-Coverage'}


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


def get_feature_df_to_boxplot(results, x_name='DS', features=['coverage', 'interval_len']):
    feature_df = deepcopy(results)[features]
    feature_df.index = [x_name + str(row_name) for row_name in feature_df.index]
    feature_df = feature_df.applymap(lambda col: {f'Experiment {i}': col[i] for i in range(len(col))})
    return feature_df.to_dict()


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


def plot_features(df, x_name='DS', x_label='DS#', limit_y=True, features=[], significance=None):
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


def summarize_col(col_series):
    corr = np.abs(col_series['test_pearson_corr'])
    col_summary = {
        'Coverage': np.mean(col_series['coverage']),
        'Interval length average': np.mean(col_series['interval_len']),
        "Pearson's correlation average": np.mean(corr),
        "log(HSIC) average": np.mean(np.log10(col_series['test_hsic'])),
        'WSC average': np.mean(col_series['test_wsc']),
        'ΔWSC average': np.mean(col_series['test_wsc_diff']),
    }
    return col_summary


def syn_summarize_col(col):
    summarized = pd.Series(summarize_col(col)).to_frame(col.name)
    n_groups = max(
        int(feature.replace("test_group_", "")[0]) for feature in col.index if feature.startswith('test_group_'))

    groups_summary = {}
    for i in range(n_groups):
        groups_summary.update({
            f'Group {i} coverage average': np.mean(col[f'test_group_{i}_coverage']),
            f'Group {i} interval length average': np.mean(col[f'test_group_{i}_interval_len']),

        })
    res = summarized.append(pd.Series(groups_summary).to_frame(col.name)).to_dict()[col.name]
    return res


def summarize_df(df, is_real):
    summarize_col_method = summarize_col if is_real else syn_summarize_col
    summary_df = pd.DataFrame(list((df.apply(summarize_col_method, axis=0)).values))
    summary_df.index = df.columns
    summary_df = summary_df.T.applymap(float_to_str)
    return summary_df


def display_cov_vs_len_graphs(col, corr_1_cov_and_len, no_corr_cov_and_len,
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


def float_to_str(n):
    if type(n) == str:
        return n

    if abs(n) < 1e-3 and abs(n) > 1e-20 and not math.isnan(n):
        if n > 0:
            sign = ''
        else:
            sign = '-'
        return sign + f'{str(n).replace(".", "").replace("0", "")[0]}e' + str(int(np.log10(abs(n))))
    else:

        if (type(n) == float and n.is_integer()) or type(n) == int:
            n_str = str(int(n))
        else:
            n_str = str(np.round(n, 3))

        return n_str


def get_delta_node_coverage(dataset_name, col_coverages, base_method_coverages,
                            length_diff, len_diff_ninth_quantiles, calibrated):
    col_node_coverages = []
    base_method_node_coverages = []
    for seed in range(len_diff_ninth_quantiles.shape[0]):
        col_node_coverage, base_method_node_coverage = \
            calculate_node_coverage(seed, dataset_name, col_coverages, base_method_coverages,
                                    length_diff, len_diff_ninth_quantiles, calibrated)
        col_node_coverages += [col_node_coverage]
        base_method_node_coverages += [base_method_node_coverage]

    col_delta_node_coverage = abs(np.array(col_node_coverages) - col_coverages.mean(axis=1))
    base_method_delta_node_coverage = abs(np.array(base_method_node_coverages) - base_method_coverages.mean(axis=1))

    return col_delta_node_coverage, base_method_delta_node_coverage


def get_delta_ILS_coverage(col_coverages, base_method_coverages,
                           length_diff, len_diff_ninth_quantiles):
    col_delta_ILS_coverage = []
    base_method_delta_ILS_coverage = []
    for seed in range(col_coverages.shape[0]):
        col_delta_ILS_coverage.append(
            abs(col_coverages[seed, :][(length_diff[seed, :].T > len_diff_ninth_quantiles[seed]).T].mean() - \
                col_coverages[seed, :].mean()))

        base_method_delta_ILS_coverage.append(
            abs(base_method_coverages[seed, :][(length_diff[seed, :].T > len_diff_ninth_quantiles[seed]).T].mean() - \
                base_method_coverages[seed, :].mean()))
    return np.array(col_delta_ILS_coverage), np.array(base_method_delta_ILS_coverage)


def add_delta_ils_and_delta_node_coverage(minority_groups_info, col, col_coverages, col_lengths, dataset_name,
                                          base_method_coverages, base_method_lengths, calibrated):
    length_diff = col_lengths - base_method_lengths
    assert length_diff.shape[1] == col_coverages.shape[1]
    len_diff_ninth_quantiles = np.sort(length_diff, axis=1)[:, int((col_coverages.shape[1]) * 0.9)]

    col_delta_node_coverage, base_method_delta_node_coverage = get_delta_node_coverage(dataset_name, col_coverages,
                                                                                       base_method_coverages,
                                                                                       length_diff,
                                                                                       len_diff_ninth_quantiles,
                                                                                       calibrated)
    minority_groups_info[col]['Delta_node_coverages_improved'] = col_delta_node_coverage
    minority_groups_info[col]['Delta_node_coverages_baseline'] = base_method_delta_node_coverage
    minority_groups_info[col][
        'Delta_node_coverage_ratio'] = col_delta_node_coverage.mean() / base_method_delta_node_coverage.mean()

    col_delta_ILS_coverage, base_method_delta_ILS_coverage = get_delta_ILS_coverage(col_coverages,
                                                                                    base_method_coverages,
                                                                                    length_diff,
                                                                                    len_diff_ninth_quantiles)
    # (OQR Delta ILS-Coverage) / (vanilla QR Delta ILS-coverage)
    minority_groups_info[col]['Delta_ILS_coverage_ratio'] = \
        col_delta_ILS_coverage.mean() / base_method_delta_ILS_coverage.mean()
    minority_groups_info[col]['Delta_ILS_coverages_improved'] = np.array(col_delta_ILS_coverage)
    minority_groups_info[col]['Delta_ILS_coverages_baseline'] = np.array(base_method_delta_ILS_coverage)


def get_minority_groups_info_and_summary_df(df, base_method, loss_method, dataset_name, desired_coverage, is_calibrated,
                                            is_real):
    base_method_column_name = get_column_name_by_method(base_method, loss=loss_method, dataset_name=dataset_name, is_real=is_real)
    base_method_column = df[base_method_column_name]
    base_method_coverages = np.array(base_method_column.loc['coverages'])
    base_method_lengths = np.array(base_method_column.loc['lengths'])

    examples_per_bin = int((1 / 100) * base_method_coverages.shape[0] * base_method_coverages.shape[1])
    base_method_cov_and_len = pd.DataFrame({'coverage': base_method_coverages.flatten(),
                                            'interval_sizes': base_method_lengths.flatten()})
    minority_groups_info = {}
    summary_df = {base_method_column_name: summarize_col(base_method_column)}
    columns = list(df.columns)
    columns.remove(base_method_column_name)
    for col in columns:
        print(col)
        summary_df[col] = summarize_col(df[col])
        col_coverages = np.array(df[col].loc['coverages']).astype(np.float32)
        col_lengths = np.array(df[col].loc['lengths']).astype(np.float32)
        col_cov_and_len = pd.DataFrame({'coverage': col_coverages.flatten(),
                                        'interval_sizes': col_lengths.flatten()})
        display_cov_vs_len_graphs(col, col_cov_and_len, base_method_cov_and_len,
                                  desired_coverage, loss_method, dataset_name, examples_per_bin)

        try:
            minority_groups_info[col] = {}
            add_delta_ils_and_delta_node_coverage(minority_groups_info, col, col_coverages, col_lengths, dataset_name,
                                                  base_method_coverages, base_method_lengths, is_calibrated)

            if len(pd.DataFrame(df).columns) == 2:  # if we compare vanilla vs orthogonal with the best corr mult
                summary_df[col]["Delta ILS-Coverage"] = minority_groups_info[col]['Delta_ILS_coverages_improved'].mean()
                summary_df[base_method_column_name]["Delta ILS-Coverage"] = minority_groups_info[col][
                    'Delta_ILS_coverages_baseline'].mean()
                summary_df[col]["Delta Node-Coverage"] = minority_groups_info[col][
                    'Delta_node_coverages_improved'].mean()
                summary_df[base_method_column_name]["Delta Node-Coverage"] = minority_groups_info[col][
                    'Delta_node_coverages_baseline'].mean()

        except Exception:
            traceback.print_exc()
            print(f"There are not enough seed results for {col}.")
            continue
    summary_df = pd.DataFrame(summary_df)
    return minority_groups_info, summary_df


def calc_std_errors(col, relevant_minorities_info=None, improved_or_baseline=None):
    sqrt_n = np.sqrt(len(col.coverage))
    col = col.copy()
    metrics = ['coverage', 'interval_len', 'test_pearson_corr', 'test_hsic', 'test_wsc_diff']

    if relevant_minorities_info is not None and improved_or_baseline is not None:
        col['Delta_ILS_coverage'] = relevant_minorities_info[f'Delta_ILS_coverages_{improved_or_baseline}']
        col['Delta_node_coverage'] = relevant_minorities_info[f'Delta_node_coverages_{improved_or_baseline}']
        metrics += ['Delta_ILS_coverage', 'Delta_node_coverage']

    mult_by_100_metrics = ['coverage', 'test_wsc_diff', 'Delta_ILS_coverage', 'Delta_node_coverage']
    for metric in [metric for metric in mult_by_100_metrics if metric in metrics]:
        assert np.max(np.abs(col[metric])) <= 1
        assert np.max(np.abs(col[metric])) <= 1
        col[metric] = np.array(col[metric]) * 100

    std_errors = {}
    std_errors['mean'] = {}
    for metric in metrics:
        std_errors[metric] = np.std(col[metric]) / sqrt_n
        std_errors['mean'][metric] = np.mean(np.abs(col[metric]))
    return std_errors

def calc_std_errors_for_two_columns(baseline_method_col, improved_method_col,
                    improved_method_column_name=None, minorities_info=None):

    if improved_method_column_name is not None and minorities_info is not None:
        relevant_minorities_info = minorities_info[improved_method_column_name]
    else:
        relevant_minorities_info = None

    base_method_std_errors = calc_std_errors(baseline_method_col, relevant_minorities_info, 'baseline')
    improved_method_std_errors = calc_std_errors(improved_method_col, relevant_minorities_info, 'improved')

    std_errors = {}
    std_errors['mean'] = {}
    for metric in [metric for metric in base_method_std_errors.keys() if metric != 'mean']:
        std_errors[metric] = [base_method_std_errors[metric],
                              improved_method_std_errors[metric]]
        std_errors['mean'][metric] = [base_method_std_errors['mean'][metric],
                                      improved_method_std_errors['mean'][metric]]
    return std_errors


def map_sign_to_color(sign):
    if sign[0] not in ['+', '-']:
        color = 'black'
    elif sign[0] == '+':
        color = 'green'
    else:
        color = 'red'
    return 'color: %s' % color


def display_final_results(loss_method, baseline_method, improved_method, is_real=True, to_latex=False):
    if is_real:
        dataset_names = ['facebook_1', 'facebook_2', 'blog_data', 'bio',
                         'kin8nm', 'naval', 'meps_19', 'meps_20', 'meps_21']
    else:
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
        if is_real:
            indexes = [name[0] for name in list(final_df.index)]
        else:
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
    except Exception as e:
        raise e
        print("No results to display.")
        return
    display(styled_final_df)
    ds_type = 'real' if is_real else 'syn'
    save_dir = f"results/final_results/{ds_type}"
    create_folder_if_it_doesnt_exist(save_dir)
    styled_final_df.to_excel(f'{save_dir}/{loss_method}, {baseline_method}_vs_{improved_method} styled_final_df.xlsx')
    if to_latex:
        final_df.index = ["\textbf{" + i + "}" for i in final_df.index]
        final_df['% Improvement'] = final_df['% Improvement'].applymap(lambda x: "\textcolor[rgb]{ 0,  .502,  0}{" + x + "}" if x[
                                                                                                  0] == '+' else "\textcolor[rgb]{ 1,  0,  0}{" + x + "}")
        latex = final_df.to_latex()
        latex = latex.replace("\\textbackslash ", "\\").replace("\{", "{").replace("\}", "}").replace("\%", "")
        print(latex)
    return styled_final_df


def get_folder_name_from_args(method_params, calibrated=False):
    method = method_params['method']
    assert method in ['QR', 'qr_forest']

    if method == 'qr_forest':
        folder_name = 'qr_forest'
    else:
        loss, corr_mult, hsic_mult = \
            method_params['loss'], method_params['corr_multiplier'], method_params['hsic_multiplier']
        corr_mult = float(corr_mult)
        hsic_mult = float(hsic_mult)
        loss = loss.replace("batch_", "")
        folder_name = f'loss=batch_{loss}_bs=1024_corr_mult={corr_mult}_hsic_mult={hsic_mult}'

    if calibrated:
        folder_name = 'cal_' + folder_name
    return folder_name


def initialize_methods_params(method, loss_method, corr_multipliers, hsic_multipliers):
    if method == 'qr_forest':
        methods_params = [{
            'method': 'qr_forest'
        }]
    else:

        if corr_multipliers is None:
            corr_multipliers = [0.0, 0.1, 0.5, 1., 3.]
        if hsic_multipliers is None:
            hsic_multipliers = [0.0, 0.1, 0.5]

        methods_params = []
        for corr in corr_multipliers:
            for hsic in hsic_multipliers:
                methods_params += [{
                    'corr_multiplier': corr,
                    'hsic_multiplier': hsic,
                    'loss': loss_method,
                    'method': 'QR'
                }]

    return methods_params


def display_results(dataset_names, method='QR', loss_method='qr',
                    corr_multipliers=None, hsic_multipliers=None, seeds=range(0, 30),
                    desired_coverage=0.9, is_real=False, is_calibrated=False,
                    base_method=VANILLA_QR, improved_method=OQR_CORR, print_latex=False,keep_cov_and_len=False):
    assert method in ['QR', 'qr_forest']
    assert (loss_method == 'qr' or loss_method == 'int' or loss_method == 'wqr') or (method != 'QR')

    methods_params = initialize_methods_params(method, loss_method, corr_multipliers, hsic_multipliers)

    std_errors = {}
    final_dataset_df = {}
    for dataset_name in dataset_names:
        res = display_results_over_dataset(dataset_name, methods_params, True, seeds,
                                           desired_coverage, is_calibrated, is_real, base_method=base_method,
                                           improved_method=improved_method)
        if 'std_errors' in res:
            std_errors[dataset_name] = res['std_errors']
        if 'final_dataset_df' in res:
            final_dataset_df[dataset_name] = res['final_dataset_df']

    if len(std_errors) == 0:
        return

    if is_calibrated:
        base_method = base_method.replace("QR", "CQR")
        improved_method = improved_method.replace("QR", "CQR")

    if loss_method == 'wqr':
        base_method = base_method.replace("QR", "WQR")
        improved_method = improved_method.replace("QR", "WQR")

    metrics = list(std_errors[dataset_names[0]].keys())
    metrics.remove("mean")
    std_errors_df = {}

    def print_method_res(dataset_name, metric):
        mean = float_to_str(std_errors[dataset_name]['mean'][metric])
        std_err = float_to_str(std_errors[dataset_name][metric])
        return f"{mean} ({std_err})"

    def print_comparison_method_res(dataset_name, metric, method_number):
        mean1 = std_errors[dataset_name]['mean'][metric][method_number]
        mean2 = std_errors[dataset_name]['mean'][metric][1 - method_number]
        std_err = std_errors[dataset_name][metric][method_number]
        if mean1 < mean2 and print_latex:
            return "\\textbf{" + float_to_str(mean1) + f" ({float_to_str(std_err)})" + "}"
        else:
            return float_to_str(mean1) + f" ({float_to_str(std_errors[dataset_name][metric][method_number])})"

    if len(methods_params) > 1 and base_method is not None and improved_method is not None:
        for metric in metrics:
            std_errors_df[metric] = {base_method: [print_comparison_method_res(dataset_name, metric, 0)
                                                   for dataset_name in dataset_names],

                                     improved_method: [print_comparison_method_res(dataset_name, metric, 1)
                                                       for dataset_name in dataset_names]
                                     }
    else:
        for metric in metrics:
            std_errors_df[metric] = [print_method_res(dataset_name, metric) for dataset_name in dataset_names]

    if not keep_cov_and_len:
        metrics.remove('coverage')
        metrics.remove('interval_len')
    if print_latex:
        bolded_dataset_names = ["\\textbf{" + dataset_name + "}" for dataset_name in dataset_names]
    else:
        bolded_dataset_names = dataset_names

    if len(methods_params) > 1 and base_method is not None and improved_method is not None:
        std_errors_df = pd.concat(
            dict(zip([metrics_rename_map[metric] for metric in metrics],
                     [pd.DataFrame(std_errors_df[metric], index=bolded_dataset_names) for metric in metrics])), axis=1)
    else:
        std_errors_df = pd.DataFrame(std_errors_df)
        std_errors_df = std_errors_df[metrics]
        std_errors_df.columns = [metrics_rename_map[metric] for metric in metrics]
        std_errors_df.index = bolded_dataset_names

    if print_latex:
        print(
            std_errors_df.to_latex(index=True).replace("\\textbackslash ", "\\").replace("\{", "{").replace("\}", "}"))

    print("Mean values and standard errors:")
    display(std_errors_df)


def get_results_dir(dataset_name, is_real, method_params):
    method_dir = get_folder_name_from_args(method_params)

    ds_type_dir = 'real_data' if is_real else 'syn_data'
    if not is_real:
        dataset_name = f'minority_group_uncertainty={dataset_name}'

    return f"{helper.results_path}{ds_type_dir}/{dataset_name}/{method_dir}"


def read_method_results(dataset_name, is_real, method_params, seeds):
    results_dir = get_results_dir(dataset_name, is_real, method_params)
    total_df = pd.DataFrame()
    for seed in seeds:
        path = f"{results_dir}/seed={seed}.csv"
        try:
            res = pd.read_csv(path)
            cov = float(str(res['coverage'].item()).replace("tensor(", "").replace(")", ""))
            res['test_wsc_diff'] = abs(float(res['test_wsc']) - cov)
            res = add_coverage_and_lengths_to_df(res)
            res = res.drop(['Unnamed: 0'], axis=1)
            total_df = total_df.append(res)
        except Exception as e:
            print(f"Didn't find results in path: {path}.")
            # raise e
            break

    return total_df


def get_dataset_name_to_display(dataset_name, is_real, shorten=False):
    if is_real:
        if shorten:
            n = ''
            if 'meps' in dataset_name:
                n = dataset_name[-2:]

            if 'facebook' in dataset_name:
                n = dataset_name[-1:]

            if 'meps' in dataset_name:
                display_dataset_name = dataset_name[0:3]
            else:
                display_dataset_name = dataset_name[0:4]

            data_txt = f"{display_dataset_name}{n}"
        else:
            data_txt = dataset_name
    else:
        data_txt = f"λ={dataset_name}"
    return data_txt


def params_to_txt(method_param, dataset_name, is_real):
    method = method_param['method']

    if method == 'qr_forest':
        param_txt = 'qr_forest'

    else:
        loss = method_param['loss']
        loss_txt = f"{loss.replace('batch_', '')}"
        corr_multiplier, hsic_multiplier = method_param['corr_multiplier'], method_param['hsic_multiplier']
        corr_multiplier = float(corr_multiplier)
        hsic_multiplier = float(hsic_multiplier)
        if hsic_multiplier == 0:
            param_txt = f'corr{corr_multiplier}'
        elif corr_multiplier == 0:
            param_txt = f'hsic{hsic_multiplier}'
        else:
            param_txt = f'corr{corr_multiplier}_hsic{hsic_multiplier}'
        param_txt = f"{loss_txt}+{param_txt}"

    data_txt = get_dataset_name_to_display(dataset_name, is_real, shorten=True)

    return f"{param_txt}_{data_txt}"


def read_results(dataset_name, is_real, methods_params, seeds):
    df = pd.DataFrame()
    for curr_params in methods_params:
        try:
            tmp_df = read_method_results(dataset_name, is_real, curr_params, seeds).applymap(lambda x: [x]).apply(
                flatten_col, axis=0)
            tmp_df.index = [params_to_txt(curr_params, dataset_name, is_real)]
            df = df.append(tmp_df)
        except Exception:
            pass

    return df


def flatten_col(col):
    return [[item for sublist in col for item in sublist]]


def get_column_name_by_method(method, dataset_name, is_real, loss=None):
    assert loss is not None  # not implemented yet
    corr_per_dataset = (real_corr_per_dataset_per_loss if is_real else syn_corr_per_dataset_per_loss)[loss]
    hsic_per_dataset = (real_hsic_per_dataset_per_loss if is_real else syn_hsic_per_dataset_per_loss)[loss]

    corr = corr_per_dataset[dataset_name] if method == OQR_CORR else 0.
    hsic = hsic_per_dataset[dataset_name] if method == OQR_HSIC else 0.
    params = {'corr_multiplier': corr, 'hsic_multiplier': hsic, 'loss': loss, 'method': 'QR'}
    method_column_name = params_to_txt(params, dataset_name, is_real)
    return method_column_name


def get_column_by_method(df, method, loss, dataset_name, is_real):
    method_column_name = get_column_name_by_method(method, dataset_name, is_real, loss)
    method_column = df[method_column_name]
    return method_column


def ratio_to_percentage_improvement(increment_ratio):
    increment_percentage = -(increment_ratio - 1) * 100
    increment_percentage_str = '{:2.2f}'.format(np.round(increment_percentage, 2))
    if increment_percentage > 0:
        increment_percentage_str = '+' + increment_percentage_str

    return increment_percentage_str


def get_improvement_df(baseline_method_column, improved_method_column, metrics):
    improvement_df = {}
    for metric in metrics:
        increment_ratio = np.mean(np.abs(improved_method_column[metric])) / \
                          np.mean(np.abs(baseline_method_column[metric]))
        improvement_df[metric] = ratio_to_percentage_improvement(increment_ratio)

    return improvement_df


def get_final_df(dataset_name, is_real, baseline_method, improved_method, baseline_method_column,
                 improved_method_column, minority_groups_info=None):
    row_name = get_dataset_name_to_display(dataset_name, is_real)

    if is_real:
        coverage_dict = {
            'Coverage': pd.DataFrame(
                {baseline_method: ['{:2.2f}%'.format(np.mean(baseline_method_column.coverage) * 100)],
                 improved_method: ['{:2.2f}%'.format(np.mean(improved_method_column.coverage) * 100)]},
                index=[row_name])
        }
        length_dict = {
            'Length': pd.DataFrame(
                {baseline_method: ['{:2.2f}'.format(np.mean(baseline_method_column.interval_len))],
                 improved_method: ['{:2.2f}'.format(np.mean(improved_method_column.interval_len))]},
                index=[row_name])
        }
    else:
        coverage_dict = {
            'Majority Coverage': pd.DataFrame({baseline_method: ['{:2.2f}%'.format(
                np.mean(baseline_method_column['test_group_0_coverage']) * 100)],
                improved_method: ['{:2.2f}%'.format(
                    np.mean(improved_method_column['test_group_0_coverage']) * 100)]},
                index=[row_name]),
            'Minority Coverage': pd.DataFrame({baseline_method: ['{:2.2f}%'.format(
                np.mean(baseline_method_column['test_group_1_coverage']) * 100)],
                improved_method: ['{:2.2f}%'.format(
                    np.mean(improved_method_column['test_group_1_coverage']) * 100)]},
                index=[row_name])
        }
        length_dict = {
            'Majority Lengths': pd.DataFrame({baseline_method: ['{:2.2f}'.format(
                np.mean(baseline_method_column['test_group_0_interval_len']))],
                improved_method: ['{:2.2f}'.format(
                    np.mean(improved_method_column['test_group_0_interval_len']))]},
                index=[row_name]),
            'Minority Lengths': pd.DataFrame({baseline_method: ['{:2.2f}'.format(
                np.mean(baseline_method_column['test_group_1_interval_len']))],
                improved_method: ['{:2.2f}'.format(
                    np.mean(improved_method_column['test_group_1_interval_len']))]},
                index=[row_name])
        }
    metrics = ['test_hsic', 'test_wsc_diff', 'test_pearson_corr']
    improvement_df = get_improvement_df(baseline_method_column, improved_method_column, metrics)
    parameters_order = ["Pearson's corr", 'HSIC', 'ΔWSC']

    add_delta_ILS_and_node = 'vanilla' in baseline_method.lower() and is_real
    if add_delta_ILS_and_node:
        assert minority_groups_info is not None
        parameters_order += ['ΔILS-Coverage', 'ΔNode-Coverage']
        for key in ['Delta_node_coverage_ratio', 'Delta_ILS_coverage_ratio']:
            improved_method_column_name = improved_method_column.name
            improvement_df[key] = ratio_to_percentage_improvement(
                minority_groups_info[improved_method_column_name][key])

    improvement_dict = {'% Improvement': pd.DataFrame(improvement_df, index=[row_name]).rename(
        columns=metrics_rename_map)[parameters_order]}
    final_dataset_dict = {**coverage_dict, **length_dict, **improvement_dict}
    final_dataset_df = pd.concat(final_dataset_dict, axis=1)
    return final_dataset_df


def compute_relative_improvement(df, dataset_name, is_real, baseline_method, improved_method,
                                 is_calibrated, loss_method=None, minority_groups_info=None):
    baseline_method_column = get_column_by_method(df, baseline_method, loss_method, dataset_name, is_real)
    improved_method_column = get_column_by_method(df, improved_method, loss_method, dataset_name, is_real)

    if is_calibrated:
        baseline_method = baseline_method.replace("QR", "CQR")
        improved_method = improved_method.replace("QR", "CQR")

    if loss_method == 'wqr':
        baseline_method = baseline_method.replace("QR", "WQR")
        improved_method = improved_method.replace("QR", "WQR")

    final_dataset_df = get_final_df(dataset_name, is_real, baseline_method, improved_method, baseline_method_column,
                                    improved_method_column, minority_groups_info)
    std_errors = calc_std_errors_for_two_columns(baseline_method_column, improved_method_column,
                                 improved_method_column_name=improved_method_column.name,
                                 minorities_info=minority_groups_info)

    if is_calibrated:
        loss_method = 'cal_' + loss_method

    save_dir = f"results/final_results/{dataset_name}"
    create_folder_if_it_doesnt_exist(save_dir)
    final_dataset_df.to_csv(
        f'{save_dir}/{loss_method} {baseline_method} vs {improved_method} final_df.csv')

    return final_dataset_df, std_errors


def display_results_over_dataset(dataset_name, methods_params, display_tables_only=False, seeds=range(30),
                                 desired_coverage=0.9, is_calibrated=False, is_real=False, base_method=VANILLA_QR,
                                 improved_method=OQR_CORR):
    if len({params['loss'] for params in methods_params if 'loss' in params}) == 1:
        loss_method = methods_params[0]['loss']
    else:
        loss_method = None

    features = ['coverage', 'interval_len', 'coverages', 'lengths',
                'test_hsic', 'test_wsc', 'test_wsc_diff',
                'test_pearson_corr', 'test_pearson_pvalue'
                ]
    if not is_real:
        n_groups = 2
        for group_number in range(n_groups):
            features += ['test_group_' + str(group_number) + '_coverage']
            features += ['test_group_' + str(group_number) + '_interval_len']

    df = read_results(dataset_name, is_real, methods_params, seeds)[features].T
    if not display_tables_only:
        df_to_plot = df.T.drop(['coverages', 'lengths'], axis=1, errors='ignore')
        plot_features(df_to_plot, x_name='', x_label='', limit_y=False, features=df_to_plot.columns)

    if is_real and len(methods_params) > 1:
        minority_groups_info, summary_df = get_minority_groups_info_and_summary_df(df, base_method, loss_method,
                                                                                   dataset_name, desired_coverage,
                                                                                   is_calibrated, is_real)
    else:
        summary_df = summarize_df(df, is_real)
        minority_groups_info = None
    # display(summary_df)

    if len(methods_params) > 1 and base_method is not None and improved_method is not None:
        try:
            final_dataset_df, std_errors = compute_relative_improvement(df, dataset_name,
                                                                        is_real,
                                                                        base_method,
                                                                        improved_method,
                                                                        is_calibrated,
                                                                        loss_method,
                                                                        minority_groups_info)
            res = {'final_dataset_df': final_dataset_df, 'std_errors': std_errors}
            return res
        except Exception:
            traceback.print_exc()
            print("Don't have the full results to construct the final comparison table.")
            return {}
    else:
        std_errors = calc_std_errors(df[df.columns[0]])
        res = {'std_errors': std_errors}
        return res