'''
Author: George Zhao
Date: 2021-11-06 09:25:33
LastEditors: George Zhao
LastEditTime: 2022-10-28 18:22:11
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import sys
sys.path.append("src")
import os
import itertools
import math
import re
import colorsys

from radarplot import *
import utils
import drawpremodel

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
import pandas as pd
import umap.umap_ as umap
from sklearn.preprocessing import MinMaxScaler

performance_labels: list = ["accuracy", "precision",
                            "f1_score", "mmc", "auc", ]

performance_labels_fancy: list = ["Accuracy", "Precision",
                                  "F1-score", "MCC", "rocAUC", ]


def get_color_bar(num: int, deep: int = 3, show=False):

    colors_list_origin = cm.get_cmap(name='hsv', lut=None)(
        np.linspace(start=0, stop=1, num=math.ceil(num / deep * 1.1))
    )

    colors_list = []
    for color in colors_list_origin:
        hsvcolor = colorsys.rgb_to_hsv(color[0], color[1], color[2])
        # Deep
        hsv_scales1 = np.linspace(start=0.75, stop=1, num=deep + 1)
        for d in range(math.floor((deep - 1) / 2)):
            colors_list.append(
                colorsys.hsv_to_rgb(
                    hsvcolor[0], hsvcolor[1], hsvcolor[2] * hsv_scales1[d])
            )
        colors_list.append(color)
        # Light
        hsv_scales2 = np.linspace(start=0.5, stop=1, num=deep + 1)
        for d in range(math.floor((deep - 1) / 2)):
            colors_list.append(
                colorsys.hsv_to_rgb(
                    hsvcolor[0], hsvcolor[1] * hsv_scales2[d], hsvcolor[2])
            )

    if show == True:
        plt.imshow([[colors_list[i] for i in range(num)], ])
        plt.show()
    return colors_list


def drawpaperfig_func(
    path_to_outdir: str,
    title: str,
    datadf: pd.DataFrame,
    colors_list: np.ndarray,
    csv_file_path: str,
):
    nrow = 2
    ncols = 2
    fig = plt.figure(constrained_layout=True,
                     figsize=(10.8 * ncols, 7.2 * nrow))
    gs = GridSpec(nrow, ncols, figure=fig)
    # Draw barplot: auc
    barplotax = fig.add_subplot(gs[0, :])
    sns.barplot(
        x=datadf.index,
        y=datadf.loc[:, 'auc'],
        ax=barplotax,
        color='#4169e1'
        # palette='b'
    )

    barplotax.set_ylim(0., 1.)
    barplotax.set_ylabel('rocAUC', fontsize=24)
    barplotax.tick_params(axis='x', rotation=90)

    # Draw radar
    # Choise the first 10 auc model.

    radar_data_col_choised = datadf.loc[:, performance_labels]
    # radar_data_choised = radar_data_col_choised.loc[[
    #     l for l in radar_data_col_choised.index if l.find('_') < 0], :]

    radar_data_choised = pd.DataFrame()

    indicator_order = ['auc', 'mmc', 'f1_score', 'precision', 'accuracy']
    for indicator_name in indicator_order:
        radar_data_col_sorted = radar_data_col_choised.sort_values(
            ([indicator_name, ] + [n for n in indicator_order if n != indicator_name]),
            ascending=[False, False, False, False, False]
        )
        row_index = 0
        while True:
            if radar_data_col_sorted.iloc[row_index, :].name in radar_data_choised.index:
                row_index += 1
                continue
            else:
                radar_data_choised = radar_data_choised.append(
                    radar_data_col_sorted.iloc[row_index, :]
                )
                break

    minest = radar_data_choised.min().min()
    maxest = radar_data_choised.max().max()
    gap_factor = 0.1
    minest_grid = math.floor(minest / gap_factor) * gap_factor
    maxest_grid = math.ceil(maxest / gap_factor) * gap_factor

    theta = radar_factory(
        radar_data_choised.shape[1],
        frame='polygon'
    )
    radarax = fig.add_subplot(
        gs[1, 0],
        projection='radar'
    )
    print(minest, "\n", maxest, "\n", minest_grid, "\n", maxest_grid, "\n", [minest_grid + gap_factor *
                                                                             i for i in range(int((maxest_grid - minest_grid) / gap_factor) + 2)])
    radarax.set_rgrids(
        [minest_grid + gap_factor *
         i for i in range(int((maxest_grid - minest_grid) / gap_factor) + 2)]
    )
    radarax.set_ylim([minest * 0.99, maxest * 1.005])

    radar_data_choised.columns = performance_labels_fancy
    for modelindex in range(radar_data_choised.shape[0]):
        radarax.plot(
            theta,
            radar_data_choised.iloc[modelindex, :],
            color=colors_list[
                list(datadf.index).index(
                    radar_data_choised.index[modelindex]
                )
            ],
            linewidth=1.5
        )
        # radarax.fill(
        #     theta,
        #     radar_data_choised.iloc[modelindex, :],
        #     color=colors_list[
        #         list(datadf.index).index(
        #             radar_data_choised.index[modelindex]
        #         )
        #     ],
        #     alpha=0.25
        # )
    radarax.set_varlabels(radar_data_choised.columns)

    legend = radarax.legend(
        radar_data_choised.index,
        loc=(1.25, 0.5),
        # loc=(0.9, .95),
        labelspacing=0.1,
        # fontsize='small'
    )

    # Draw UMAP
    featuredatadf = pd.read_csv(csv_file_path, index_col=0)
    scaler = MinMaxScaler()
    scaler.fit(featuredatadf)
    featuredatadf = scaler.transform(featuredatadf)
    umaper = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        n_epochs=10000,
        min_dist=0.1,
        local_connectivity=1,
        random_state=1
    )
    z1 = umaper.fit_transform(featuredatadf)
    df1 = pd.DataFrame()
    df1["UMAP-1"] = z1[:, 0]
    df1["UMAP-2"] = z1[:, 1]
    df1["Tag"] = list(itertools.chain(
        ['T', ] * int(featuredatadf.shape[0] / 2),
        ['N', ] * int(featuredatadf.shape[0] / 2),
    ))
    sns.scatterplot(
        x="UMAP-1",
        y="UMAP-2",
        hue="Tag",
        # style="predlable",
        hue_order=['T', 'N'],
        # style_order=['T', 'N'],
        # palette="hls",
        data=df1,
        ax=fig.add_subplot(gs[1, 1])
    )

    fig.tight_layout()
    # fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(path_to_outdir, f"{title}.pdf"))
    plt.close(fig)


if __name__ == '__main__':
    import argparse
    import argcomplete
    parser = argparse.ArgumentParser(prog='libradarplot')
    subparsers = parser.add_subparsers(dest='subparser')
    test_subparser = subparsers.add_parser('test')
    test_subparser.set_defaults(func=lambda _: 'test_subparser')

    drawresult_subparser = subparsers.add_parser('drawpremodel')
    drawresult_subparser.add_argument('--modeldir', type=str, required=True,
                                      help='Path to Model Parent Dir.')
    drawresult_subparser.add_argument('--title', type=str, required=True,
                                      help='Title.')
    drawresult_subparser.add_argument(
        '--optimal', action='store_true', help='Optimal the Model result.')
    drawresult_subparser.add_argument(
        '--cv', action='store_true', help='CV or TT.')
    drawresult_subparser.add_argument(
        '--sub', action='store_true', help='Sub Model.')
    drawresult_subparser.add_argument('--outputdir', type=str, required=True,
                                      help='Path to Output dir.')
    drawresult_subparser.set_defaults(func=lambda _: 'drawresult_subparser')

    drawpaperfig_subparser = subparsers.add_parser('drawpaperfig')
    drawpaperfig_subparser.add_argument('--modeldir', type=str, required=True,
                                        help='Path to Model Parent Dir.')
    drawpaperfig_subparser.add_argument('--title', type=str, required=True,
                                        help='Title.')
    drawpaperfig_subparser.add_argument(
        '--optimal', action='store_true', help='Optimal the Model result.')
    drawpaperfig_subparser.add_argument(
        '--cv', action='store_true', help='CV or TT.')
    drawpaperfig_subparser.add_argument(
        '--sub', action='store_true', help='Sub Model.')
    drawpaperfig_subparser.add_argument('--outputdir', type=str, required=True,
                                        help='Path to Output dir.')
    drawpaperfig_subparser.add_argument('--csvpath', type=str, required=True,
                                        help='Path to csvpath.')
    drawpaperfig_subparser.add_argument('--color_model_list', type=str, required=True,
                                        help='Path to color_model_list.')
    drawpaperfig_subparser.set_defaults(
        func=lambda _: 'drawpaperfig_subparser')

    drawcolor_subparser = subparsers.add_parser('drawcolor')
    drawcolor_subparser.add_argument('-n', type=int, required=True,
                                     help='Num.')
    drawcolor_subparser.set_defaults(
        func=lambda _: 'drawcolor_subparser')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.func(args) == 'drawresult_subparser' or args.func(args) == 'drawpaperfig_subparser':
        if os.path.exists(args.outputdir) == False:
            os.makedirs(args.outputdir)
        modeldirname_list = [
            dir_path
            for dir_path in os.listdir(args.modeldir)
            if os.path.isdir(os.path.join(args.modeldir, dir_path))
            and os.path.exists(os.path.join(os.path.join(args.modeldir, dir_path), ".reportignore")) == False
        ]
        modeldir_list = None
        if args.sub == False:
            modeldir_list = [
                os.path.join(
                    os.path.join(args.modeldir, dir_path),
                    (
                        "cv_model.json"
                        if args.cv == True else
                        "tt_model.json"
                    )
                )
                for dir_path in modeldirname_list
            ]
        else:
            modeldir_list = list(itertools.chain(
                *[
                    [
                        os.path.join(
                            os.path.join(args.modeldir, dir_path),
                            filename
                        )
                        for filename in os.listdir(os.path.join(args.modeldir, dir_path))
                        if re.match(
                            '.*cv_model.json' if args.cv == True else ".*tt_model.json",
                            filename
                        ) is not None
                    ]
                    for dir_path in modeldirname_list
                ]
            ))
        data_name = utils.model_reporter.load_name(
            path_to_json_list=modeldir_list,
        )
        data_mat = utils.model_reporter.load_ds(
            path_to_json_list=modeldir_list,
            index_=None if args.cv == True else 0,
            optimal=args.optimal
        )
        data_mat[data_mat < 0] = 0
        datadf = pd.DataFrame(
            data_mat, index=data_name, columns=utils.model_reporter.col_name
        )
        # colors_list = cm.get_cmap(name='hsv', lut=None)(
        #     np.linspace(start=0, stop=1, num=data_mat.shape[0] + 1))
        model_to_color = None
        with open(args.color_model_list, "r", encoding='UTF-8') as _color_model_list_f:
            model_to_color = sorted(
                list(set(_color_model_list_f.read().splitlines())))

        model_to_color_list = get_color_bar(len(model_to_color))

        colors_list = ["#000000", ] * data_mat.shape[0]
        for model_name, color in zip(model_to_color, model_to_color_list):
            colors_list[data_name.tolist().index(model_name)] = color

        if args.func(args) == 'drawresult_subparser':
            drawpremodel.plot_model_result(
                path_to_outdir=args.outputdir,
                title=f"{args.title}" + "_o"if args.optimal == True else "",
                case_data=data_mat,
                labels=data_name,
            )
            drawpremodel.plot_displot_model_result(
                path_to_outdir=args.outputdir,
                title=f"{args.title}_Displot" +
                "_o"if args.optimal == True else "",
                case_data=data_mat,
                labels=data_name,
            )
        elif args.func(args) == 'drawpaperfig_subparser':
            drawpaperfig_func(
                path_to_outdir=args.outputdir,
                title=f"{args.title}_fig" +
                "_o"if args.optimal == True else "",
                datadf=datadf,
                colors_list=colors_list,
                csv_file_path=args.csvpath
            )
        else:
            parser.print_help()

    elif args.func(args) == 'test_subparser':
        data = [['Sulfate', 'Nitrate', 'EC', 'OC1', 'OC2', 'OC3', 'OP', 'CO', 'O3'],
                ('Basecase', [
                    [0.88, 0.01, 0.03, 0.03, 0.00, 0.06, 0.01, 0.00, 0.00],
                    [0.07, 0.95, 0.04, 0.05, 0.00, 0.02, 0.01, 0.00, 0.00],
                    [0.01, 0.02, 0.85, 0.19, 0.05, 0.10, 0.00, 0.00, 0.00],
                    [0.02, 0.01, 0.07, 0.01, 0.21, 0.12, 0.98, 0.00, 0.00],
                    [0.01, 0.01, 0.02, 0.71, 0.74, 0.70, 0.00, 0.00, 0.00]])]

        N = len(data[0])
        theta = radar_factory(N, frame='polygon')

        spoke_labels = data.pop(0)
        title, case_data = data[0]

        fig, ax = plt.subplots(
            figsize=(6, 6), subplot_kw=dict(projection='radar'))
        fig.subplots_adjust(top=0.85, bottom=0.05)

        ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
        ax.set_title(title, position=(0.5, 1.1), ha='center')

        for d in case_data:
            line = ax.plot(theta, d)
            ax.fill(theta, d, alpha=0.25)
        ax.set_varlabels(spoke_labels)

        plt.show()
    elif args.func(args) == 'drawcolor_subparser':
        get_color_bar(args.n, show=True)
    else:
        parser.print_help()
