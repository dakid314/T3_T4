'''
Author: George Zhao
Date: 2022-08-15 19:44:24
LastEditors: George Zhao
LastEditTime: 2022-08-15 19:46:49
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import sys
sys.path.append("src")
import os
import math

from radarplot import *
import utils

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import pandas as pd

plot_sub_model = False


def plot_model_result(
    path_to_outdir: str,
    title: str,
    case_data: np.ndarray,
    labels: list,
    csv: bool = True,
    spoke_labels: list = ["accuracy", "precision",
                          "f1_score", "mmc", "auc", ],
    spoke_labels_fancy: list = ["Accuracy", "Precision",
                                "F1-score", "MCC", "rocAUC", ]
):
    # path_to_outdir: str, ## Output dir path
    # title: str, ## the title of figure
    # case_data: np.ndarray, ## the table of performance score
    # labels: list,  ## model name list.
    # csv: bool = True, ## save csv?

    # Clean and declare data.
    spoke_labels_origin = utils.model_reporter.col_name

    case_data[case_data < 0] = 0

    if plot_sub_model == False:
        # save submodel csv when not plot submodel
        df = pd.DataFrame(case_data, columns=spoke_labels_origin,
                          index=labels)
        if csv != False:
            path_to_csv = os.path.join(path_to_outdir, f"{title}.sub.csv")
            path_to_json = os.path.join(path_to_outdir, f"{title}.sub.json")
            df.to_csv(path_to_csv, index_label='index_col')
            df.to_json(path_to_json,)
        # Filt submodel
        df = df.loc[[l for l in labels if l.find('_') < 0], :]
        case_data = df.values
        labels = df.index

    # Plot radar
    theta = radar_factory(
        len(spoke_labels),
        frame='polygon'
    )

    colors_list = cm.get_cmap(name='rainbow_r', lut=None)(
        np.linspace(start=0, stop=1, num=case_data.shape[0] + 1))

    fig, ax = plt.subplots(
        figsize=(10.8, 5.4),
        subplot_kw=dict(projection='radar')
    )
    fig.subplots_adjust(top=0.85, bottom=0.05)
    ax.set_rgrids([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title(title, position=(0.5, 1.1), ha='center')

    case_data_choise = case_data[:, [
        spoke_labels_origin.index(
            spoke_labels[spoke_labels_index_origin]
        )
        for spoke_labels_index_origin in range(len(spoke_labels))
    ]]

    for d, c in zip(case_data_choise, colors_list):
        line = ax.plot(theta, d, color=c)
        ax.fill(theta, d, color=c, alpha=0.25)
    ax.set_varlabels(spoke_labels)

    legend = ax.legend(
        labels,
        loc=(1.2, 0.),
        # loc=(0.9, .95),
        labelspacing=0.1,
        fontsize='small'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_outdir, f"{title}.pdf"))
    if csv != False:
        path_to_csv = os.path.join(path_to_outdir, f"{title}.csv")
        path_to_json = os.path.join(path_to_outdir, f"{title}.json")
        df = pd.DataFrame(case_data, columns=spoke_labels_origin,
                          index=labels)
        df.to_csv(path_to_csv, index_label='index_col')
        df.to_json(path_to_json,)
    pass


def plot_displot_model_result(
    path_to_outdir: str,
    title: str,
    case_data: np.ndarray,
    labels: list,
    spoke_labels: list = ["accuracy", "precision",
                          "f1_score", "mmc", "auc", ],
    spoke_labels_fancy: list = ["Accuracy", "Precision",
                                "F1-score", "MCC", "rocAUC", ]
):
    # path_to_outdir: str, ## Output dir path
    # title: str, ## the title of figure
    # case_data: np.ndarray, ## the table of performance score
    # labels: list,  ## model name list.

    # Clean and declare data.
    spoke_labels_origin = utils.model_reporter.col_name
    if spoke_labels is None:
        spoke_labels = spoke_labels_origin
    case_data[case_data < 0] = 0

    # colors_list for all model(labels)
    colors_list = cm.get_cmap(name='rainbow_r', lut=None)(
        np.linspace(start=0, stop=1, num=case_data.shape[0] + 1))

    nrow = math.ceil(len(spoke_labels) / 2)
    ncols = 2
    fig, axs = plt.subplots(
        nrows=nrow,
        ncols=ncols,
        figsize=(10.8 * ncols, 7.2 * nrow),
    )

    if nrow * ncols > len(spoke_labels):
        for i in range(1, (nrow * ncols - len(spoke_labels)) + 1):
            fig.delaxes(axs[nrow - 1][ncols - i])

    # Iter spoke_labels (performence)
    for spoke_labels_index_origin in range(len(spoke_labels)):
        spoke_labels_index = spoke_labels_origin.index(
            spoke_labels[spoke_labels_index_origin]
        )
        ax = axs[math.floor(spoke_labels_index / 2), spoke_labels_index % 2]
        sns.barplot(
            x=labels,
            y=case_data[:, spoke_labels_index],
            ax=ax,
            palette=colors_list
        )

        ax.set_ylim(0., 1.)
        ax.set_ylabel(spoke_labels_fancy[spoke_labels_index], fontsize=24)
        ax.tick_params(axis='x', rotation=90)
    # fig.suptitle(f'{title}', ha='center', fontsize=36)
    fig.tight_layout()
    # fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(path_to_outdir, f"{title}.pdf"))
    pass
