'''
Author: George Zhao
Date: 2021-08-04 17:20:28
LastEditors: George Zhao
LastEditTime: 2022-08-13 22:12:12
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import os
import re
import json
import itertools
import math

import pandas as pd
import numpy as np

from . import ds_preprocess

std_switch = False

_score_list = ["accuracy", "precision", "f1_score", "mmc", "auc",
               "specificity", "sensitivity", {"confusion_matrix": ["tn", "tp", "fn", "fp"]}]

col_name = []
for score_key in _score_list:
    if isinstance(score_key, str) == True:
        col_name.append(score_key)
    elif isinstance(score_key, dict) == True:
        col_name.extend(
            itertools.chain(*[
                [
                    k
                    for k in score_key[key]
                ] for key in score_key
            ])
        )


def get_detail(result_json_premodel: dict, optimal: bool, _score_list: list = _score_list, ):

    if optimal == True:
        dict_ = [result_json_premodel, ]
        fpr, tpr, thresholds = ds_preprocess.roc_curve(
            result_json_premodel['testing']['origin']['label'], result_json_premodel['testing']['origin']['pred'])
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        dict_ = ds_preprocess.Five_Cross_Evaluation(
            dict_, optimal_threshold)
        result_json_premodel = dict_[0]

    result = []
    for score_key in _score_list:
        if isinstance(score_key, str) == True:
            result.append(
                result_json_premodel['testing']['evaluation'][score_key])
        elif isinstance(score_key, dict) == True:
            result.extend(
                itertools.chain(*[
                    [
                        result_json_premodel['testing']['evaluation'][key][k]
                        for k in score_key[key]
                    ] for key in score_key
                ])
            )
    return result


def load_ds(
    path_to_json_list: list,
    index_: int,
    optimal: bool = False
):
    # Make Sure that All position in the table are point to the same Protein.
    # Feature
    db = None
    feature = list()
    for path_to_json in path_to_json_list:
        with open(path_to_json, 'r', encoding='UTF-8') as f:
            db = json.load(f)
        if index_ is not None:
            feature.append(
                get_detail(result_json_premodel=db[index_], optimal=optimal,)
            )
        else:
            # get Average.
            result = []
            for ith in range(len(db)):
                result.append(get_detail(
                    result_json_premodel=db[ith], optimal=optimal,
                ))
            feature.append(np.array(result).sum(axis=0) / len(db))
    return np.array(feature)


def load_name(
    path_to_json_list: list,
):
    # Make Sure that All position in the table are point to the same Protein.
    # Feature
    db = None
    feature = list()
    for path_to_json in path_to_json_list:
        with open(path_to_json, 'r', encoding='UTF-8') as f:
            db = json.load(f)
        feature.append(
            db[0]['detail']['model']
        )
    return np.array(feature)


def get_md_report(path_to_root: str,
                  path_to_out: str = None,
                  desc: str = '',
                  optimal: bool = False
                  ):
    path_to_root = os.path.join(path_to_root, '')
    path_stack = [path_to_root, ]
    path_of_result_list = list()
    while len(path_stack) != 0:
        path_current = path_stack.pop()
        if os.path.isdir(path_current) == True:
            if os.path.exists(
                os.path.join(path_current, '.reportignore')
            ) == False:
                path_stack.extend([os.path.join(path_current, item)
                                   for item in os.listdir(path_current)])
            continue
        else:
            # With Traning Data.
            # reresult = re.findall(
            #     r'^(.+_result\.json)$', path_current)
            # if len(reresult) != 1:
            #     continue
            # else:
            #     path_of_result_list.append(reresult[0])
            reresult = re.findall(
                r'^(.+_model\.json)$', path_current)
            if len(reresult) != 1:
                reresult = re.findall(
                    r'^(.+T\dstack.+_model\.json)$', path_current)
                if len(reresult) != 1:
                    continue
                else:
                    path_of_result_list.append(reresult[0])
            else:
                path_of_result_list.append(reresult[0])

    report_content_index = []
    report_content_index_model = []
    report_content = []
    for resultjson_path in path_of_result_list:
        result_json_dict = None
        with open(resultjson_path, 'r', encoding='UTF-8') as resultjson_f:
            result_json_dict = json.load(resultjson_f)

        resultjson_path = os.path.splitext(resultjson_path)[0]
        if resultjson_path[-6:] == '_model':
            resultjson_path = resultjson_path[:-6]
        if optimal == True:
            resultjson_path = resultjson_path + '_o'
            for i in range(len(result_json_dict)):
                dict_ = [result_json_dict[i], ]
                fpr, tpr, thresholds = ds_preprocess.roc_curve(
                    result_json_dict[i]['testing']['origin']['label'], result_json_dict[i]['testing']['origin']['pred'])
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
                dict_ = ds_preprocess.Five_Cross_Evaluation(
                    dict_, optimal_threshold)
                result_json_dict[i] = dict_[0]

        for index, result_json_premodel in enumerate(result_json_dict):
            # ! get_detail() From Here.
            result = []
            for score_key in _score_list:
                if isinstance(score_key, str) == True:
                    result.append(
                        result_json_premodel['testing']['evaluation'][score_key])
                elif isinstance(score_key, dict) == True:
                    result.extend(
                        itertools.chain(*[
                            [
                                result_json_premodel['testing']['evaluation'][key][k]
                                for k in score_key[key]
                            ] for key in score_key
                        ])
                    )
            report_content.append(result)
            report_content_index.append(resultjson_path)
            report_content_index_model.append(index)

        # Average Calculation
        num_of_model = int(report_content_index_model[-1]) + 1
        if num_of_model > 1:
            last_profile_of_model = report_content[-1 * num_of_model:]
            average_vector = [
                sum([last_profile_of_model[i][j]
                     for i in range(num_of_model)]) / num_of_model
                for j in range(len(last_profile_of_model[0]))
            ]

            report_content_index.append(resultjson_path)
            report_content_index_model.append('Average')
            report_content.append(average_vector)

            if std_switch == True:
                # STD Calculation
                if num_of_model <= 1:
                    std_vector = [0., ] * len(last_profile_of_model[0])
                else:
                    std_vector = [
                        math.sqrt(
                            sum(
                                [
                                    pow(
                                        (
                                            last_profile_of_model[i][j]
                                            -
                                            average_vector[j]
                                        ), 2
                                    )
                                    for i in range(num_of_model)
                                ]
                            ) / (num_of_model - 1)
                        )
                        for j in range(len(last_profile_of_model[0]))
                    ]
                report_content_index.append(resultjson_path)
                report_content_index_model.append('STD')
                report_content.append(std_vector)

    # return [
    #     p[len(path_to_root) + 1:]
    #     for p in path_of_result_list
    # ], report_content_index_model, report_content
    df = pd.DataFrame(report_content)
    df.columns = col_name
    # df.set_index([[
    #     p[len(path_to_root) + 1:]
    #     for p in path_of_result_list
    # ], report_content_index_model])
    df['index'] = [
        p[len(path_to_root):]
        for p in report_content_index
    ]
    df['index_sub'] = report_content_index_model
    df.set_index(['index', 'index_sub'], inplace=True)

    if os.path.exists(os.path.split(path_to_out)[0]) == False:
        os.makedirs(os.path.split(path_to_out)[0])

    if isinstance(path_to_out, str):
        if os.path.splitext(path_to_out)[1] == '.md':
            df.to_markdown(path_to_out)
        elif os.path.splitext(path_to_out)[1] == '.csv':
            df.to_csv(path_to_out)
        elif os.path.splitext(path_to_out)[1] == '.xlsx':
            df.to_excel(path_to_out)
        elif os.path.splitext(path_to_out)[1] == '.html':
            df.to_html(path_to_out)
        else:
            pass
        return None
    else:
        return df
