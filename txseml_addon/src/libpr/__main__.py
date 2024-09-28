'''
Author: George Zhao
Date: 2022-08-19 23:07:50
LastEditors: George Zhao
LastEditTime: 2022-11-12 17:01:20
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
'''
Author: George Zhao
Date: 2021-11-06 09:25:33
LastEditors: George Zhao
LastEditTime: 2022-08-19 20:12:32
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
import datetime
import json

import utils
import libmodel

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
import pandas as pd
import umap.umap_ as umap
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve

performance_labels: list = ["accuracy", "precision",
                            "f1_score", "mmc", "auc", "specificity", "sensitivity", 'fp']

performance_labels_fancy: list = ["Accuracy", "Precision",
                                  "F1-score", "MCC", "rocAUC", "Specficity", "Senstivity", "FPR"]


def load_voting_name(
    path_to_json_list: list,
):
    # Make Sure that All position in the table are point to the same Protein.
    # Feature
    db = None
    feature = list()
    for path_to_json in path_to_json_list:
        with open(path_to_json, 'r', encoding='UTF-8') as f:
            db = json.load(f)
        name = f"{db[0]['detail']['model']} {db[0]['detail']['desc']}"
        if 'sp' in db[0]['detail']['opt'].keys():
            name += f" {db[0]['detail']['opt']['sp']}"
        feature.append(
            name
        )
    return np.array(feature)


if __name__ == '__main__':
    import argparse
    import argcomplete
    parser = argparse.ArgumentParser(prog='libradarplot')
    subparsers = parser.add_subparsers(dest='subparser')

    votetest_subparser = subparsers.add_parser('report')
    votetest_subparser.add_argument('--modeldir', type=str, required=True,
                                    help='Path to Model Parent Dir.')
    votetest_subparser.add_argument('--outputdir', type=str, required=True,
                                    help='Path to Output dir.')
    votetest_subparser.add_argument('--desc', type=str, required=True,
                                    help='Desc.')
    votetest_subparser.set_defaults(
        func=lambda _: 'report_subparser')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.func(args) == 'report_subparser':
        # No optimal default.

        result_list = []
        for file in os.listdir(args.modeldir):
            if os.path.isfile(os.path.join(args.modeldir, file)) and re.match(r'.+\.json', file):
                result_list.append(os.path.join(args.modeldir, file))
        # Load ########################################
        data_name = load_voting_name(
            path_to_json_list=result_list,
        )
        data_mat = utils.model_reporter.load_ds(
            path_to_json_list=result_list,
            index_=0,
            optimal=False
        )

        # ROCauc Plot
        # Load data
        roc_df_list = list()
        for json_file_path in result_list:
            with open(json_file_path, 'r', encoding='UTF-8') as jsonf:
                dict_ = json.load(jsonf)

                precision, recall, thresholds = precision_recall_curve(
                    dict_[0]['testing']['origin']['label'],
                    dict_[0]['testing']['origin']['pred']
                )
                _model_name_ = f"{dict_[0]['detail']['model']}_{dict_[0]['detail']['desc']}"
                _model_name_ += " (area = {:.3f})".format(
                    auc(recall, precision))
                roc_df_list.append(
                    pd.DataFrame({
                        'recall': recall,
                        'precision': precision,
                        'Model': _model_name_
                    })
                )
        # Plot data
        fig = plt.figure(constrained_layout=True,
                         figsize=(10.8, 7.2))
        gs = GridSpec(1, 1, figure=fig)
        lineplotax = fig.add_subplot(gs[:])
        concanteddf = pd.concat(roc_df_list)
        concanteddf = concanteddf.reset_index(drop=True)
        for _df in roc_df_list:
            lineplotax.plot(_df.loc[:, 'recall'], _df.loc[:, 'precision'],
                            label=_df.at[0, "Model"])
        lineplotax.set_xlabel("Recall")
        lineplotax.set_ylabel("Precision")
        lineplotax.set_xlim([-0.05, 1.05])
        lineplotax.set_ylim([-0.05, 1.05])
        lineplotax.legend(loc=4)
        lineplotax.set_aspect('equal', 'box')
        plt.savefig(
            os.path.join(
                args.outputdir, f'{args.desc}-voting-prc-{datetime.datetime.now().strftime("%Y%h%d%H%M%S")}.pdf')
        )
        plt.close()

    else:
        parser.print_help()
