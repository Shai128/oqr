import numpy as np
import pandas as pd
import sklearn

from helper import get_x_test


def calculate_node_coverage(seed, dataset_name, with_corr_coverages, no_corr_coverages,
                            length_diff, ninth_quantile, calibrated=False):
    x = get_x_test(dataset_name, seed)
    if calibrated:
        idx = np.arange(len(x))  # np.random.permutation(len(y_upper))
        n_half = int(np.floor(len(x) / 2))
        idx_test, _ = idx[:n_half], idx[n_half:2 * n_half]
        x = x[idx_test]
        length_diff = length_diff[:, idx_test]
        with_corr_coverages = with_corr_coverages[:, idx_test]
        no_corr_coverages = no_corr_coverages[:, idx_test]

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
    no_corr_coverage_in_node = no_corr_coverages[seed][decision_path[:, best_node] == 1].mean()

    return with_corr_coverage_in_node, no_corr_coverage_in_node