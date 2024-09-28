'''
Author: George Zhao
Date: 2021-07-19 23:49:59
LastEditors: George Zhao
LastEditTime: 2022-06-26 15:18:50
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import pandas as pd
import numpy as np
import math
import json
from sklearn.metrics import accuracy_score, precision_score, f1_score, matthews_corrcoef
from sklearn.metrics import roc_curve, auc, confusion_matrix

mtkl_5CV_mode = False


def make_oneline_df(df, col=None):
    dfn = pd.DataFrame(df).T
    dfn.columns = col
    return dfn


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def make_merge(t_p_f: pd.DataFrame, t_p_l: np.array, t_n_f: pd.DataFrame, t_n_l: np.array):
    t_f = pd.concat([t_p_f, t_n_f])
    t_l = np.concatenate([t_p_l, t_n_l])

    return t_f, t_l


def get_5C_data(
    shape_to_chiose: int,
    t_p_f: pd.DataFrame,
    t_p_l: np.array,
    t_n_f: pd.DataFrame,
    t_n_l: np.array,
    shufflesplit=None
):
    if shufflesplit is None:
        # 5-Cross
        shape_of_each_part = math.floor(shape_to_chiose / 5)
        Five_split = list()
        for i in range(5):
            start = shape_of_each_part * i
            end = (i + 1) * shape_of_each_part
            if shape_to_chiose - end < shape_of_each_part:
                end = shape_to_chiose
            Five_split.append(
                (
                    (
                        t_p_f.iloc[start:end, :],
                        t_p_l[start:end]
                    ),
                    (
                        t_n_f.iloc[start:end, :],
                        t_n_l[start:end]
                    )
                )
            )
        Five_Cross = list()
        for i in range(5):
            Five_Cross.append(
                (
                    [
                        pd.concat(
                            [
                                pd.concat([Five_split[j][0][0]
                                           for j in range(5) if j != i], ignore_index=False),
                                pd.concat([Five_split[j][1][0]
                                           for j in range(5) if j != i], ignore_index=False),
                            ]
                        ),
                        np.concatenate(
                            [
                                np.concatenate([Five_split[j][0][1]
                                                for j in range(5) if j != i]),
                                np.concatenate([Five_split[j][1][1]
                                                for j in range(5) if j != i]),
                            ]
                        )
                    ],
                    [
                        pd.concat([Five_split[i][0][0], Five_split[i]
                                   [1][0]], ignore_index=False),
                        np.concatenate(
                            [Five_split[i][0][1], Five_split[i][1][1]])
                    ]
                )
            )
        return Five_Cross
    else:
        if mtkl_5CV_mode == True:
            # TODO Policy 2
            Five_Cross = list()
            shufflesplit_index_dict = None
            with open(shufflesplit['shufflesplit_index_file'], 'r', encoding='UTF-8') as f:
                shufflesplit_index_dict = json.load(f)

            # v_p_f = shufflesplit_index_dict['v_p_f']
            # v_p_l = shufflesplit_index_dict['v_p_l']
            # v_n_f = shufflesplit_index_dict['v_n_f']
            # v_n_l = shufflesplit_index_dict['v_n_l']

            for shufflesplit_list_index in range(shufflesplit_index_dict['option']['n_split']):
                Five_Cross.append(
                    [
                        [
                            # t
                            # f
                            pd.concat([
                                t_p_f.iloc[shufflesplit_index_dict['t']
                                           ['p']['t'][shufflesplit_list_index], :],
                                t_n_f.iloc[shufflesplit_index_dict['t']
                                           ['n']['t'][shufflesplit_list_index], :]
                            ], ignore_index=False),
                            # l
                            np.concatenate([
                                t_p_l[shufflesplit_index_dict['t']['p']
                                      ['t'][shufflesplit_list_index]],
                                t_n_l[shufflesplit_index_dict['t']['n']
                                      ['t'][shufflesplit_list_index]]
                            ])
                        ],
                        [
                            # v
                            # f
                            pd.concat([
                                t_p_f.iloc[shufflesplit_index_dict['t']
                                           ['p']['v'][shufflesplit_list_index], :],
                                t_n_f.iloc[shufflesplit_index_dict['t']
                                           ['n']['v'][shufflesplit_list_index], :]
                            ], ignore_index=False),
                            # l
                            np.concatenate([
                                t_p_l[shufflesplit_index_dict['t']['p']
                                      ['v'][shufflesplit_list_index]],
                                t_n_l[shufflesplit_index_dict['t']['n']
                                      ['v'][shufflesplit_list_index]]
                            ])
                        ]
                    ]
                )
            return Five_Cross
        Five_Cross = Five_Cross = list()
        t_f, t_l = make_merge(
            t_p_f=t_p_f,
            t_p_l=t_p_l,
            t_n_f=t_n_f,
            t_n_l=t_n_l
        )
        t_f.reset_index(drop=True, inplace=True)
        for i in range(t_f.shape[0]):
            Five_Cross.append(
                (
                    (
                        t_f.drop(index=i),
                        np.delete(t_l, i, 0)
                    ),
                    (
                        make_oneline_df(t_f.iloc[i, :], col=t_f.columns),
                        np.array([t_l[i], ])
                    ),
                )
            )
        return Five_Cross


def Five_Cross_Evaluation(model_result_set: list, pro_cutoff: float, mode: str = None):
    if mtkl_5CV_mode == False and mode == 'loo':
        new_result = {}
        new_result['detail'] = model_result_set[0]['detail']
        new_result['testing'] = {
            'origin': {
                'pred': [model_result_set[i]['testing']['origin']['pred'][0] for i in range(len(model_result_set))],
                'label': [model_result_set[i]['testing']['origin']['label'][0] for i in range(len(model_result_set))]
            },
            'evaluation': {},
            'option': {}
        }
        model_result_set = [new_result, ]

    for i in range(len(model_result_set)):
        for t_V in model_result_set[i].keys():
            if t_V == 'detail':
                continue
            pred = model_result_set[i][t_V]['origin']['pred']
            pred_l = [1 if i >= pro_cutoff else 0 for i in pred]
            label = model_result_set[i][t_V]['origin']['label']
            fpr, tpr, _ = roc_curve(label, pred)
            confusion_matrix_1d = confusion_matrix(label, pred_l).ravel()
            confusion_dict = {N: n for N, n in zip(['tn', 'fp', 'fn', 'tp'], list(
                confusion_matrix_1d * 2 / np.sum(confusion_matrix_1d)))}
            model_result_set[i][t_V]['evaluation'].update(
                {
                    "accuracy": accuracy_score(label, pred_l),
                    "precision": precision_score(label, pred_l),
                    "f1_score": f1_score(label, pred_l),
                    "mmc": matthews_corrcoef(label, pred_l),
                    "auc": auc(fpr, tpr),
                    "specificity": confusion_dict['tn'] / (confusion_dict['tn'] + confusion_dict['fp']),
                    "sensitivity": confusion_dict['tp'] / (confusion_dict['tp'] + confusion_dict['fn']),
                    "confusion_matrix": confusion_dict,
                    "_roc_Data": {'fpr': list(fpr), 'tpr': list(tpr)}
                }
            )
            if 'option' not in model_result_set[i][t_V]:
                model_result_set[i][t_V]['option'] = {}
            model_result_set[i][t_V]['option'].update(
                {'pro_cutoff': pro_cutoff}
            )
    return model_result_set


def make_binary_label(size: int, label: bool):
    if label == True:
        return np.ones(shape=(size,))
    else:
        return np.zeros(shape=(size,))


def merge_pd_list(pd_list: list):
    return pd.concat(pd_list, axis=1)


def consturct_vertor(i, t=20):
    v = np.zeros(shape=(t))
    if i is None:
        pass
    else:
        v[i] = 1
    return v


def make_feature_dividend_list(f_length_list):
    result = [0, ]
    for element_length in f_length_list:
        result.append(result[-1] + element_length)
    return result
