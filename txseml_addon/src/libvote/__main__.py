'''
Author: George Zhao
Date: 2022-08-19 23:07:50
LastEditors: George Zhao
LastEditTime: 2022-09-03 17:08:03
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


def load_voting_cutoff(
    path_list: list,
    optimal: bool = True,
    sp: float = None
):
    # Make Sure that All position in the table are point to the same Protein.
    # Feature
    feature = list()
    for path_ in path_list:
        with open(path_, 'r', encoding='UTF-8') as f:
            db = json.load(f)
            if optimal == True:
                fpr, tpr, thresholds = utils.ds_preprocess.roc_curve(
                    db[0]['testing']['origin']['label'], db[0]['testing']['origin']['pred'])
                if sp is None:
                    optimal_idx = np.argmax(tpr - fpr)
                    optimal_threshold = thresholds[optimal_idx]
                else:
                    rocdf = pd.DataFrame({
                        "fpr": fpr,
                        "tpr": tpr,
                        "thresholds": thresholds,
                        "nfpr": np.abs(fpr - 1 + sp)
                    }).sort_values(
                        ["nfpr", "tpr"], ascending=[True, False]
                    )
                    optimal_threshold = rocdf.iloc[0, :].at["thresholds"]

                feature.append(
                    optimal_threshold
                )
            else:
                feature.append(
                    db[0]['testing']['option']['pro_cutoff']
                )
    return np.array(feature)


def load_pred_data(path_list: list, model_name: list, p_l: str = 'pred', train_or_test: str = 'testing'):
    result_data = []
    for path_ in path_list:
        with open(path_, 'r', encoding='UTF-8') as jsonf:
            _data = json.load(jsonf)[0][train_or_test]['origin'][p_l]
            if type(_data[0]) is list:
                _data = [i[0] for i in _data]
            result_data.append(_data)
    return pd.DataFrame(result_data, index=model_name)


if __name__ == '__main__':
    import argparse
    import argcomplete
    parser = argparse.ArgumentParser(prog='libradarplot')
    subparsers = parser.add_subparsers(dest='subparser')

    votetest_subparser = subparsers.add_parser('libvote')
    votetest_subparser.add_argument('--modeldir', type=str, required=True,
                                    help='Path to Model Parent Dir.')
    votetest_subparser.add_argument('--sp', type=float, default=0.95,
                                    help='sensitivity.')
    votetest_subparser.add_argument('--votingsp', type=float, required=False,
                                    help='sensitivity of voting.')
    votetest_subparser.add_argument('--desc', type=str, required=True,
                                    help='desc.')
    votetest_subparser.add_argument('--start', type=int, help='start rank.')
    votetest_subparser.add_argument('--end', type=int, help='end rank.')
    votetest_subparser.add_argument(
        '--optimal', action='store_true', help='Optimal the Model result.')
    votetest_subparser.add_argument(
        '--sub', action='store_true', help='Sub Model.')
    votetest_subparser.add_argument('--outputdir', type=str, required=True,
                                    help='Path to Output dir.')
    votetest_subparser.set_defaults(
        func=lambda _: 'votetest_subparser')

    votetest_subparser = subparsers.add_parser('report')
    votetest_subparser.add_argument('--modeldir', type=str, required=True,
                                    help='Path to Model Parent Dir.')
    votetest_subparser.add_argument('--outputdir', type=str, required=True,
                                    help='Path to Output dir.')
    votetest_subparser.add_argument('--desc', type=str, required=True,
                                    help='Desc.')
    votetest_subparser.set_defaults(
        func=lambda _: 'report_subparser')

    subparser = subparsers.add_parser('cutoffsuball')
    subparser.add_argument('--sp', type=float, default=0.95,
                           help='sensitivity.')
    subparser.add_argument('--modeldir', type=str, required=True,
                           help='Path to Model Parent Dir.')
    subparser.add_argument('--outputdir', type=str, required=True,
                           help='Path to outputdir.')
    subparser.add_argument('--desc', type=str, required=True,
                           help='desc.')
    subparser.set_defaults(
        func=lambda _: 'cutoffsuball_subparser')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.func(args) == 'cutoffsuball_subparser':

        modeldirname_list = [
            dir_path
            for dir_path in os.listdir(args.modeldir)
            if os.path.isdir(os.path.join(args.modeldir, dir_path))
            and os.path.exists(os.path.join(os.path.join(args.modeldir, dir_path), ".reportignore")) == False
        ]

        modeldir_list = list(itertools.chain(
            *[
                [
                    os.path.join(
                        os.path.join(args.modeldir, dir_path),
                        filename
                    )
                    for filename in os.listdir(os.path.join(args.modeldir, dir_path))
                    if re.match(
                        ".*tt_model.json",
                        filename
                    ) is not None
                ]
                for dir_path in modeldirname_list
            ]
        ))
        data_name = utils.model_reporter.load_name(
            path_to_json_list=modeldir_list,
        )
        # get First or end rank model name ######################################
        model_to_vote_name = data_name
        model_name_path_dict = dict(zip(data_name, modeldir_list))
        # Get Model Path ########################################################
        model_to_vote = [
            model_name_path_dict[model_name] for model_name in model_to_vote_name
        ]
        thresholds_dict = {}
        for model_index in range(len(model_to_vote)):
            bestone_dict = None
            with open(model_to_vote[model_index], 'r', encoding='UTF-8') as f:
                bestone_dict = json.load(f)
            bestone_fpr, bestone_tpr, bestone_thresholds = utils.ds_preprocess.roc_curve(
                bestone_dict[0]['testing']['origin']['label'], bestone_dict[0]['testing']['origin']['pred']
            )
            if args.sp is None:
                best_one_optimal_idx = np.argmax(bestone_tpr - bestone_fpr)
                bestone_optimal_threshold = bestone_thresholds[best_one_optimal_idx]
            else:
                bestone_rocdf = pd.DataFrame({
                    "fpr": bestone_fpr,
                    "tpr": bestone_tpr,
                    "thresholds": bestone_thresholds,
                    "nfpr": np.abs(bestone_fpr - 1 + args.sp)
                }).sort_values(
                    ["nfpr", "tpr"], ascending=[True, False]
                )
                bestone_optimal_threshold = bestone_rocdf.iloc[
                    0, :].at["thresholds"]
            thresholds_dict[model_to_vote_name[model_index]
                            ] = bestone_optimal_threshold

        with open(os.path.join(args.outputdir, f"T{args.desc}.subcutoff.{args.sp if args.sp is not None else 'nosp'}.json"), 'w+', encoding='UTF-8') as f:
            json.dump(thresholds_dict, f)

    elif args.func(args) == 'report_subparser':
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
        data_mat[data_mat < 0] = 0
        # xlsx add best sub plot (added in libvote)
        datadf = pd.DataFrame(
            data_mat, index=data_name, columns=utils.model_reporter.col_name
        )

        datadf = datadf.loc[:, performance_labels]
        datadf.columns = performance_labels_fancy

        datadf.to_excel(os.path.join(
            args.outputdir, f'{args.desc}-voting-{datetime.datetime.now().strftime("%Y%h%d%H%M%S")}.xlsx'))
        # ROCauc Plot
        # Load data
        roc_df_list = list()
        for json_file_path in result_list:
            with open(json_file_path, 'r', encoding='UTF-8') as jsonf:
                dict_ = json.load(jsonf)
                _model_name_ = f"{dict_[0]['detail']['model']}_{dict_[0]['detail']['desc']}"
                _model_name_ += " (area = {:.3f})".format(
                    dict_[0]['testing']['evaluation']['auc'])
                roc_df_list.append(
                    pd.DataFrame({
                        'fpr': dict_[0]['testing']['evaluation']['_roc_Data']['fpr'],
                        'tpr': dict_[0]['testing']['evaluation']['_roc_Data']['tpr'],
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
            lineplotax.plot(_df.loc[:, 'fpr'], _df.loc[:, 'tpr'],
                            label=_df.at[0, "Model"])
        lineplotax.set_xlabel("1 - Specificity")
        lineplotax.set_ylabel("Sensitivity")
        lineplotax.set_xlim([-0.05, 1.05])
        lineplotax.set_ylim([-0.05, 1.05])
        lineplotax.legend(loc=4)
        lineplotax.set_aspect('equal', 'box')
        plt.savefig(
            os.path.join(
                args.outputdir, f'{args.desc}-voting-roc-{datetime.datetime.now().strftime("%Y%h%d%H%M%S")}.pdf')
        )
        plt.close()

    elif args.func(args) == 'votetest_subparser':
        if args.votingsp is None:
            args.votingsp = args.sp
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
                    "tt_model.json"
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
                            ".*tt_model.json",
                            filename
                        ) is not None
                    ]
                    for dir_path in modeldirname_list
                ]
            ))
        data_name = utils.model_reporter.load_name(
            path_to_json_list=modeldir_list,
        )

        # performance Load #######################################################
        data_mat = utils.model_reporter.load_ds(
            path_to_json_list=modeldir_list,
            index_=0,
            optimal=args.optimal
        )
        data_mat[data_mat < 0] = 0
        datadf = pd.DataFrame(
            data_mat, index=data_name, columns=utils.model_reporter.col_name
        )

        # Get Sorted. ############################################################
        model_name_path_dict = dict(zip(data_name, modeldir_list))
        model_to_vote_name_sorted = datadf.sort_values(
            ['auc', ], ascending=[False, ]
        )

        # Remove Same origin one. ###############################################
        model_name_sortedbyauc = list(model_to_vote_name_sorted.index)
        model_name_sortedbyauc_remove_hemomodel = list()
        origin_name_list_exist = list()
        for model_name_ in model_name_sortedbyauc:
            origin_modol_name = model_name_.split('_')[0]
            if origin_modol_name not in origin_name_list_exist:
                origin_name_list_exist.append(origin_modol_name)
                model_name_sortedbyauc_remove_hemomodel.append(model_name_)

        # get First or end rank model name ######################################
        model_to_vote_name = model_to_vote_name_sorted.loc[
            model_name_sortedbyauc_remove_hemomodel, :].iloc[args.start:args.end, :].index

        # Get Model Path ########################################################
        model_to_vote = [
            model_name_path_dict[model_name] for model_name in model_to_vote_name
        ]

        # Load trainging ########################################################
        traning_feature = load_pred_data(
            model_to_vote, model_to_vote_name, 'pred', 'training').T
        training_label = load_pred_data(
            model_to_vote, model_to_vote_name, 'label', 'training').iloc[0, :]

        # selector = SelectKBest(f_classif, k=5)
        # selector.fit(traning_feature, training_label)
        # feature = feature.iloc[:, selector.get_support(indices=True)]

        # Load testing ##########################################################
        # Feature: [[1,0,0,0,1],[1,0,0,0,1]]
        # Row: seq: [1,0,0,1]
        # Col: model: [1,0,0,1]
        # label_m: [[xxxxxxxxxx],[xxxxxxxx]]
        # Row: model: []
        # Col: seq: []
        feature = load_pred_data(
            model_to_vote, model_to_vote_name, 'pred', 'testing').T  # sum in axis=1 means get
        label = load_pred_data(
            model_to_vote, model_to_vote_name, 'label', 'testing').iloc[0, :]  # get seq label

        # Get pred #############################################################
        submodel_cutoff = load_voting_cutoff(
            path_list=model_to_vote,
            sp=args.sp
        )
        cutoffed_df = (feature >= submodel_cutoff)
        pred_f = cutoffed_df.sum(axis=1).values / cutoffed_df.shape[1]

        opt_dict = args.__dict__
        del opt_dict['func']
        result_dict = {
            "testing": {
                "origin": {
                    f'pred': pred_f.tolist(),
                    f'label': label.values.tolist()
                },
                "evaluation": {
                },
                "option": {
                }
            },
            "detail": {
                "model": 'Voting',
                'desc': args.desc,
                'opt': opt_dict,
                "model_list": list(feature.columns),
                "submodel_cutoff": submodel_cutoff.tolist()}
        }

        dict_ = [result_dict, ]
        fpr, tpr, thresholds = utils.ds_preprocess.roc_curve(
            result_dict['testing']['origin']['label'], result_dict['testing']['origin']['pred'])
        # For voting
        if args.votingsp is None:
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
        else:
            rocdf = pd.DataFrame({
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thresholds,
                "nfpr": np.abs(fpr - 1 + args.votingsp)
            }).sort_values(
                ["nfpr", "tpr"], ascending=[True, False]
            )
            optimal_threshold = rocdf.iloc[0, :].at["thresholds"]

        dict_ = utils.ds_preprocess.Five_Cross_Evaluation(
            dict_,
            pro_cutoff=optimal_threshold,
            mode=None
        )
        dict_[0]['testing']['option']['optimal'] = True
        with open(os.path.join(args.outputdir, *[f'{args.desc}.json', ]), 'w+', encoding='UTF-8') as f:
            json.dump(dict_, f, cls=utils.ds_preprocess.MyEncoder)

        # Save bestmodel performance
        bestone_dict = None
        with open(model_to_vote[0], 'r', encoding='UTF-8') as f:
            bestone_dict = json.load(f)
        bestone_fpr, bestone_tpr, bestone_thresholds = utils.ds_preprocess.roc_curve(
            label, feature.loc[:, model_to_vote_name[0]]
        )
        if args.votingsp is None:
            best_one_optimal_idx = np.argmax(bestone_tpr - bestone_fpr)
            bestone_optimal_threshold = bestone_thresholds[best_one_optimal_idx]
        else:
            bestone_rocdf = pd.DataFrame({
                "fpr": bestone_fpr,
                "tpr": bestone_tpr,
                "thresholds": bestone_thresholds,
                "nfpr": np.abs(bestone_fpr - 1 + args.votingsp)
            }).sort_values(
                ["nfpr", "tpr"], ascending=[True, False]
            )
            bestone_optimal_threshold = bestone_rocdf.iloc[
                0, :].at["thresholds"]
        bestone_dict = utils.ds_preprocess.Five_Cross_Evaluation(
            bestone_dict,
            pro_cutoff=bestone_optimal_threshold,
            mode=None
        )
        bestone_dict[0]['testing']['option']['optimal'] = True
        bestone_dict[0]['detail']['opt'] = opt_dict
        with open(os.path.join(
            args.outputdir, *[
                f'bestone_{bestone_dict[0]["detail"]["model"]}.json',
            ]
        ), 'w+', encoding='UTF-8') as f:
            json.dump(bestone_dict, f, cls=utils.ds_preprocess.MyEncoder)

    else:
        parser.print_help()
