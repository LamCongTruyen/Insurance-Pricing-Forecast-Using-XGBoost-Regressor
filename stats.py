from itertools import combinations, product
import pandas as pd
import numpy as np
import scipy.stats as stats


def chi2(X, correction=True):
    col_pairs = list(combinations(X.columns, 2))
    results_list = []
    for col1, col2 in col_pairs:
        contingency = pd.crosstab(
            X[col1],
            X[col2]
        )
        chi2, p_val, dof, exp_freq = stats.chi2_contingency(
            contingency.values, correction=correction
        )
        results_list.append([col1, col2, chi2, p_val, dof])
    results = pd.DataFrame(
        results_list,
        columns=[
            'column1', 'column2', 'chi_squared', 'p_value', 'dof'
        ]
    )
    return results


# Gọi hàm chi2 và in kết quả

def anova(X):
    num_cols = X.select_dtypes(np.number).columns
    cat_cols = X.select_dtypes(object).columns
    col_pairs = list(product(num_cols, cat_cols))
    results_list = []
    for num_col, cat_col in col_pairs:
        X_filtered_list = []
        cat_values = X[cat_col].unique()
        for cat_value in cat_values:
            X_filtered_list.append(X[X[cat_col] == cat_value][num_col].values)
        f_stat, p_val = stats.f_oneway(*X_filtered_list)
        results_list.append([num_col, cat_col, f_stat, p_val])
    results = pd.DataFrame(
        results_list,
        columns=[
            'num_column', 'cat_column', 'f_stat', 'p_value'
        ]
    )
    print(results)
    return results
