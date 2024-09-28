import os
import json
import gzip
import pickle
import itertools

import numpy as np
import pandas as pd
import openpyxl

import tqdm

from bayes_opt import BayesianOptimization


def generate_bit_combinations(bit_len: int):
    # 生成初始列表
    list1 = [0, ] * bit_len
    # 遍历所有可能的组合
    for i in range(2**len(list1)):
        # 将 i 转换为二进制字符串，并去掉前面的 '0b' 标记
        bin_str = bin(i)[2:]
        # 将字符串左侧填充 0，直到长度与 list1 相同
        bin_str = bin_str.zfill(len(list1))
        # 将字符串中的每个字符分别转换为 int，替换 list1 中对应的元素
        for j in range(len(list1)):
            list1[j] = int(bin_str[j])
        # 使用 yield 将当前状态作为生成器的返回值
        yield list1


def merge_pn_dataset(
    p_f: pd.DataFrame,
    p_l: np.ndarray,
    n_f: pd.DataFrame,
    n_l: np.ndarray
):
    t_f = pd.concat([p_f, n_f])
    t_l = np.concatenate([p_l, n_l])

    return t_f, t_l


from . import model
from . import model_space


class FeatureGroupSelection:
    def __init__(self, desc: str = "Default") -> None:

        self.feature_order = None
        self.search_result_df = None
        self.desc = desc
        self.search_mode = os.environ['SEARCH_MODE']
        pass

    def search(
        self,
        ordered_data: list,
        path_to_save_dir: str,
        n_jobs: int,
        feature_prob: float = 0.5
    ):
        assert n_jobs > 0
        # data = [
        #     {
        #         "t_p": "xxx",
        #         "t_n": "xxx",
        #         "v_p": "xxx",
        #         "v_n": "xxx",
        #         "name": "name"
        #     }, {}, {}, {}
        # ]
        self.feature_order = [item["name"] for item in ordered_data]

        assert len(set(self.feature_order)) == len(ordered_data)

        os.makedirs(
            f"{path_to_save_dir}/{self.search_mode}",
            exist_ok=True
        )

        with open(f"{path_to_save_dir}/{self.search_mode}/feature_order.json", "w+", encoding='UTF-8') as f:
            json.dump(self.feature_order, f)

        self.search_result_df = pd.DataFrame()
        # for feature_scheme_index, feature_select_vec in tqdm.tqdm(
        #     enumerate(generate_bit_combinations(bit_len=len(ordered_data))),
        #     total=2**len(ordered_data)
        # ):

        def target_func(**kwargs):
            feature_select_vec = [
                1 if kwargs[f"{k}"] <= feature_prob else 0
                for k in range(len(kwargs.keys()))
            ]
            if sum(feature_select_vec) == 0:
                return np.nan

            feature_select_vec_bits_str = ''.join(map(str, feature_select_vec))
            feature_select_int = int(feature_select_vec_bits_str, 2)

            feature_selected = list(itertools.compress(
                ordered_data, feature_select_vec
            ))
            feature_selected_name = [item["name"] for item in feature_selected]

            # 合并feature_selected成数据集
            data_set_split = {
                datatype: pd.concat([
                    item[datatype] for item in feature_selected
                ], axis=1)
                for datatype in ["t_p", "t_n", "v_p", "v_n"]
            }
            label_set_split = {
                datatype: np.ones(
                    shape=(feature_selected[0][datatype].shape[0], ))
                for datatype in ["t_p", "v_p",]
            } | {
                datatype: np.zeros(
                    shape=(feature_selected[0][datatype].shape[0], ))
                for datatype in ["t_n", "v_n",]
            }

            t_f, t_l = merge_pn_dataset(
                p_f=data_set_split["t_p"],
                p_l=label_set_split["t_p"],
                n_f=data_set_split["t_n"],
                n_l=label_set_split["t_n"],
            )
            v_f, v_l = merge_pn_dataset(
                p_f=data_set_split["v_p"],
                p_l=label_set_split["v_p"],
                n_f=data_set_split["v_n"],
                n_l=label_set_split["v_n"],
            )

            # 丢入model中进行训练
            os.makedirs(
                f"{path_to_save_dir}/{self.search_mode}/{feature_select_int}",
                exist_ok=True
            )

            search_result_in_a_scheme_df = pd.DataFrame()
            for model_index in range(len(model_space.find_space)):
                model_information_summary, searched_result_performance_summary, searched_result_5C_performance_summary = model.MyOptimitzer(
                    classifier_name=model_space.find_space[model_index]['name'],
                    classifier_class=model_space.find_space[model_index]['class'],
                    classifier_param_dict=model_space.find_space[model_index]['param'],
                ).find_best(
                    X=t_f.values,
                    y=t_l,
                    validation=(v_f.values, v_l),
                    search_method=(
                        "BayesSearchCV"
                        if "Bayes" not in model_space.find_space[model_index]
                        or model_space.find_space[model_index]['Bayes'] == True
                        else "GridSearchCV"
                    ),
                    n_jobs=n_jobs
                ).get_summary(
                    # 保存为 {path_to_save_dir}/{self.search_mode}/{feature_select_int}/xxxmodel.pkl(.pdf)
                    path_to_dir=f"{path_to_save_dir}/{self.search_mode}/{feature_select_int}"
                )

                # 记录结果，插入到 search_result_in_a_scheme_df
                result_series = pd.concat([
                    pd.Series(model_information_summary),
                    pd.Series(feature_select_vec, index=self.feature_order),
                    pd.Series({"Total_Feature": sum(feature_select_vec), }),
                    pd.Series(searched_result_performance_summary),
                    pd.Series(searched_result_5C_performance_summary),
                ], keys=[
                    "Model_Information",
                    "Feature_Selected",
                    "Total_Feature",
                    "Best_Performance",
                    "5FoldCV_Performance",
                ])

                result_series.name = (feature_select_int, model_index)

                search_result_in_a_scheme_df = pd.concat([
                    search_result_in_a_scheme_df,
                    result_series.to_frame().T
                ], axis=0, ignore_index=False)

                search_result_in_a_scheme_df.index = search_result_in_a_scheme_df.index.set_names(
                    ["Feature_Scheme", "Model_Type"]
                )

                local_xlsx_path = f"{path_to_save_dir}/{self.search_mode}/{feature_select_int}/searched_result.xlsx"

                # 缓存 search_result_in_a_scheme_df
                search_result_in_a_scheme_df.to_excel(
                    local_xlsx_path,
                    self.desc,
                    freeze_panes=(3, 2)
                )

            # 记录局部结果，插入到 self.search_result_df
            self.search_result_df = pd.concat([
                self.search_result_df,
                search_result_in_a_scheme_df
            ], axis=0, ignore_index=False)

            self.search_result_df.index = self.search_result_df.index.set_names(
                ["Feature_Scheme", "Model_Type"]
            )

            # 缓存 self.search_result_df
            total_xlsx_path = f"{path_to_save_dir}/{self.search_mode}/searched_result.xlsx"
            self.search_result_df.to_excel(
                total_xlsx_path,
                self.desc,
                freeze_panes=(3, 2)
            )
            return search_result_in_a_scheme_df.loc[:, [['Best_Performance', 'rocAUC'],]].max().item()

        if self.search_mode == "Bayes":
            bayes_optimizer = BayesianOptimization(
                f=target_func,
                pbounds={
                    f"{data_index}": [0, 1]
                    for data_index in range(len(ordered_data))
                },
                random_state=42
            )

            bayes_optimizer.maximize(
                init_points=128,
                n_iter=256,
            )
        elif self.search_mode == "Onehot":
            for feature_index in range(len(ordered_data)):
                target_func(**{
                    f"{data_index}": 0 if feature_index == data_index else 1
                    for data_index in range(len(ordered_data))
                })

        # with gzip.open(f"{path_to_save_dir}/{self.search_mode}/bayes_optimizer.pkl", "wb") as f:
        #     pickle.dump(bayes_optimizer, f)

        return self

    def order_feature(self, data):

        assert len(set([item["name"] for item in data])) == len(data)

        assert self.feature_order is None

        data_dict = {
            item["name"]: item
            for item in data
        }

        ordered_data = [
            data_dict[feature_name]
            for feature_name in self.feature_order
        ]

        return ordered_data
