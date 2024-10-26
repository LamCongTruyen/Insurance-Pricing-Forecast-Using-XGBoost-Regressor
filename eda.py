from plotly.subplots import make_subplots
import plotly.graph_objects as go
from itertools import combinations, product
import pandas as pd
import numpy as np
import plotly.express as px


def plot_histograms(X, height=1200):
    fig = make_subplots(rows=X.shape[1], cols=1)
    for i, col in enumerate(X.columns):
        fig.add_trace(
            go.Histogram(
                x=X[col],
                name=col,
            ),
            row=i+1,
            col=1
        )
        fig.update_xaxes(
            title_text=col,
            row=i+1,
            col=1
        )
        fig.update_yaxes(
            title_text='count',
            row=i+1,
            col=1
        )
    fig.update_layout(height=height)
    fig.show()


def plot_univariate_numeric(X, y):
    fig = make_subplots(rows=X.shape[1], cols=1)
    for i, col in enumerate(X.columns):
        fig.add_trace(
            go.Scatter(
                x=X[col],
                y=y,
                name=col,
                mode='markers'
            ),
            row=i+1,
            col=1
        )
        fig.update_xaxes(
            title_text=col,
            row=i+1,
            col=1
        )
        fig.update_yaxes(
            title_text='charges',
            row=i+1,
            col=1
        )
    fig.update_layout(
        height=1200
    )
    fig.show()


def plot_univariate_categorical(X, y):
    fig = make_subplots(rows=X.shape[1], cols=1)
    for i, col in enumerate(X.columns):
        fig.add_trace(
            go.Box(
                x=X[col],
                y=y,
                name=col,
            ),
            row=i+1,
            col=1
        )
        fig.update_xaxes(
            title_text=col,
            row=i+1,
            col=1
        )
        fig.update_yaxes(
            title_text='charges',
            row=i+1,
            col=1
        )
    fig.update_layout(height=1200)
    fig.show()


def plot_heatmap(X, y, bins=10):
    data = pd.concat([X, y], axis=1)
    for num_col in X.select_dtypes(include=np.number):
        if X[num_col].nunique() < bins:
            continue
        else:
            data[num_col] = pd.cut(data[num_col], bins=bins)
    col_pairs = list(combinations(X.columns, 2))
    for col1, col2 in col_pairs:
        col_pair_y_mean = data.groupby(
            [col1, col2]
        )[y.name].mean().reset_index()
        col_pair_y_mean = col_pair_y_mean.pivot(
            index=col1, columns=col2, values=y.name
        )
        col_pair_y_mean.sort_index(ascending=False, inplace=True)
        col_pair_y_mean.index = col_pair_y_mean.index.astype(str)
        col_pair_y_mean.columns = col_pair_y_mean.columns.astype(str)
        fig = px.imshow(col_pair_y_mean)
        fig.show()


def plot_paired_boxplots(X, y):
    col_pairs = list(combinations(X.columns, 2))
    fig = make_subplots(rows=len(col_pairs), cols=1)
    for i, (col1, col2) in enumerate(col_pairs):
        paired_cat = col1 + '=' + X[col1] + ', ' + col2 + '=' + X[col2]
        fig.add_trace(
            go.Box(
                x=paired_cat,
                y=y,
                name=f'{col1} & {col2}'
            ),
            row=i+1,
            col=1
        )
        fig.update_xaxes(
            title_text=f'{col1} & {col2}',
            row=i+1,
            col=1,
            categoryorder='array',
            categoryarray=sorted(paired_cat.unique())
        )
        fig.update_yaxes(
            title_text='charges',
            row=i+1,
            col=1
        )
    fig.update_layout(
        height=1800
    )
    fig.show()


def plot_paired_scatterplots(X, y):
    data = pd.concat([X, y], axis=1)
    num_cols = X.select_dtypes(np.number).columns
    cat_cols = X.select_dtypes(object).columns
    col_pairs = list(product(num_cols, cat_cols))
    fig = make_subplots(rows=len(col_pairs), cols=1)
    j = 1
    for i, (col1, col2) in enumerate(col_pairs):
        for col2_val in X[col2].unique():
            mask = X[col2] == col2_val
            X_ = X[mask]
            y_ = y[mask]
            fig.add_trace(
                go.Scatter(
                    # data_frame=data,
                    x=X_[col1],
                    y=y_,
                    # marker_color=X_[col2],
                    name=col2_val,
                    mode='markers',
                    legendgroup=j
                ),
                row=i+1,
                col=1
            )
        fig.update_xaxes(title_text=col1, row=i+1, col=1)
        fig.update_yaxes(title_text=y.name, row=i+1, col=1)
        j += 1
    fig.update_layout(
        height=4800,
        legend_tracegroupgap=485,
    )
    fig.show()


def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig = px.histogram(residuals)
    fig.show()


def plot_pearson_wrt_target(X, y):
    data = pd.concat([X, y], axis=1)
    data_corr = data.select_dtypes(np.number).corr()
    data_corr.index.name = 'features'
    data_corr = data_corr.reset_index()
    data_corr = data_corr[data_corr['features'] != y.name]
    fig = px.bar(
        data_frame=data_corr.reset_index(),
        x='features',
        y='charges'
    )
    fig.update_yaxes(title='correlation')
    fig.show()
