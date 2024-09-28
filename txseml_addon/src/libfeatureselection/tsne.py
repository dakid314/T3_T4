import os
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['pdf.use14corefonts'] = False
# mpl.rcParams['pdf.usecorefonts'] = True
mpl.rcParams['pdf.compression'] = 9

import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'nature'])

import seaborn as sns

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import umap.umap_ as umap

from sklearn.preprocessing import MinMaxScaler

n_jobs = (
    max(1, os.cpu_count() - 2)
    if "n_jobs" not in os.environ or os.environ['n_jobs'] == "" else
    int(os.environ['n_jobs'])
)


def feature_2d_plot(
    data: np.ndarray,
    label: np.ndarray,
    desc: str,
    path_to_out_dir: str,
    n_jobs: int = n_jobs
):
    # desc Need Type and Feature

    os.makedirs(path_to_out_dir, exist_ok=True)

    ground_true_label_list = label

    data = MinMaxScaler().fit_transform(data)

    tsne = TSNE(
        n_components=2,
        verbose=0,
        n_jobs=n_jobs,
        random_state=42
    )
    z0 = MinMaxScaler().fit_transform(tsne.fit_transform(data))

    umaper = umap.UMAP(
        n_neighbors=5,
        n_components=2,
        n_epochs=10000,
        min_dist=0.1,
        local_connectivity=1,
        n_jobs=n_jobs,
        random_state=42
    )
    z1 = MinMaxScaler().fit_transform(umaper.fit_transform(data))

    df0 = pd.DataFrame()
    df0["comp-1"] = z0[:, 0]
    df0["comp-2"] = z0[:, 1]
    df0["truelabel"] = [
        'T' if item ==
        1 else 'N' for item in ground_true_label_list
    ]

    df1 = pd.DataFrame()
    df1["comp-1"] = z1[:, 0]
    df1["comp-2"] = z1[:, 1]
    df1["truelabel"] = [
        'T' if item ==
        1 else 'N' for item in ground_true_label_list
    ]

    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(19.2 / 2, 10.8 / 4),
    )
    sns.scatterplot(
        x="comp-1",
        y="comp-2",
        hue="truelabel",
        # style="predlable",
        hue_order=['T', 'N'],
        # style_order=['T', 'N'],
        # palette="hls",
        data=df0,
        ax=ax[0]
    ).set(title=f"{desc} T-SNE projection")
    ax[0].set_aspect('equal', adjustable='box')
    sns.scatterplot(
        x="comp-1",
        y="comp-2",
        hue="truelabel",
        # style="predlable",
        hue_order=['T', 'N'],
        # style_order=['T', 'N'],
        # palette="hls",
        data=df1,
        ax=ax[1]
    ).set(title=f"{desc} UMAP projection")
    ax[1].set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(os.path.join(path_to_out_dir, f"{desc}_tsne.pdf"))
    plt.close(fig)
    return


def feature_2d_plot_for_dataset(
    dataset: list,
    path_to_out_dir: str,
    n_jobs: int = n_jobs
):
    # dataset = [
    #     {
    #         "t_p": "xxx",
    #         "t_n": "xxx",
    #         "v_p": "xxx",
    #         "v_n": "xxx",
    #         "name": "name"
    #     }, {}, {}, {}
    # ]
    for feature_set in dataset:
        p_f = pd.concat([
            feature_set['t_p'], feature_set['v_p']
        ])
        p_l = np.ones((p_f.shape[0], ))
        n_f = pd.concat([
            feature_set['t_n'], feature_set['v_n']
        ])
        n_l = np.zeros((n_f.shape[0], ))

        f = pd.concat([
            p_f, n_f
        ])
        l = np.concatenate([
            p_l, n_l
        ])

        feature_2d_plot(
            data=f,
            label=l,
            desc=feature_set['name'],
            path_to_out_dir=path_to_out_dir,
            n_jobs=n_jobs
        )
