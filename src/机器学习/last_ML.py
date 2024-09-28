from sklearn.model_selection import StratifiedKFold
import os
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
import os
import typing
from datetime import datetime

from sklearn.base import ClassifierMixin

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import GridSearchCV, StratifiedKFold,StratifiedShuffleSplit
from sklearn.model_selection._search import BaseSearchCV
from skopt import BayesSearchCV

import numpy as np
import pandas as pd
import pickle
import gzip
import pymrmr
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.calibration import CalibratedClassifierCV

from skopt.space import Real, Categorical
from skopt import BayesSearchCV
from sklearn.model_selection import cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import roc_curve, confusion_matrix, precision_score, accuracy_score, f1_score, matthews_corrcoef, auc,precision_recall_curve

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import warnings
warnings.filterwarnings("ignore")

def get_evaluation(label: list, pred: list, pro_cutoff: float = None):
    fpr, tpr, thresholds = roc_curve(label, pred)
    if pro_cutoff is None:
        best_one_optimal_idx = np.argmax(tpr - fpr)
        pro_cutoff = thresholds[best_one_optimal_idx]
    pred_l = [1 if i >= pro_cutoff else 0 for i in pred]
    confusion_matrix_1d = confusion_matrix(label, pred_l).ravel()
    confusion_dict = {N: n for N, n in zip(['tn', 'fp', 'fn', 'tp'], list(
        confusion_matrix_1d * 2 / np.sum(confusion_matrix_1d)))}
    
    precision, recall, _ = precision_recall_curve(label, pred)
    pr_auc = auc(recall, precision)
    
    evaluation = {
        "accuracy": accuracy_score(label, pred_l),
        "precision": precision_score(label, pred_l),
        "f1_score": f1_score(label, pred_l),
        "mmc": matthews_corrcoef(label, pred_l),
        "rocAUC": auc(fpr, tpr),
        "prAUC": pr_auc,
        "specificity": confusion_dict['tn'] / (confusion_dict['tn'] + confusion_dict['fp']),
        "sensitivity": confusion_dict['tp'] / (confusion_dict['tp'] + confusion_dict['fn']),
        'pro_cutoff': pro_cutoff
    }
    return evaluation

def plot_roc_curve(target, pred, path_to_: str):
    fpr, tpr, thresholds = roc_curve(target, pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(19.2, 10.8))
    plt.plot(fpr, tpr, color='red', lw=2,
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) curve')
    plt.legend(loc="lower right")

    plt.savefig(f"{path_to_}")
    plt.clf()

    
feature_list = ['18pp','AAC','BPBaac','CTDC','CTDT','CTriad','onehot',
                'PC-PseAAC','ppt25','QSO','SC-PseAAC','CTDD','DPC']#'ppt','18pp'

bac_name_list=['30_Acinetobacter_baumannii_K09_14.fasta','30_Advenella_kashmirensis_WT001.fasta','30_Agrobacterium_tumefaciens_12D1.fasta',
               '30_Escherichia_coli_MG1655.fasta','30_Klebsiella_pneumoniae_subsp.pneumoniae_HS11286.fasta','30_Stenotrophomonas_sp.CD2.fasta']

for feature_name in feature_list:
    for bac_name in bac_name_list:
        if feature_name in ['SC-PseAAC', 'PC-PseAAC']:
            neg_df = pd.read_csv(f'/mnt/md0/Public/T3_T4/txseml_addon/out/libfeatureselection/bac_neg/{bac_name}_{feature_name}.csv', header=None)
            pos_df = pd.read_csv(f'/mnt/md0/Public/T3_T4/txseml_addon/out/libfeatureselection/30_feature_research_pos/T3_training_30.fasta_{feature_name}.csv', header=None)
        else:
            neg_df = pd.read_csv(f'/mnt/md0/Public/T3_T4/txseml_addon/out/libfeatureselection/bac_neg/{bac_name}_{feature_name}.csv')
            pos_df = pd.read_csv(f'/mnt/md0/Public/T3_T4/txseml_addon/out/libfeatureselection/30_feature_research_pos/T3_training_30.fasta_{feature_name}.csv')
        if feature_name == 'BPBaac':
            feature = pd.read_csv(f'/mnt/md0/Public/T3_T4/txseml_addon/out/libfeatureselection/bac_neg/{bac_name}_{feature_name}.csv')
            feature = feature.iloc[0:,1:]
        elif feature_name in ['PC-PseAAC','SC-PseAAC']:
            pos_df1 = pos_df.iloc[0:,0:]
            neg_df1 = neg_df.iloc[0:,0:]
        elif feature_name in ['CTriad','onehot','QSO']:
            pos_df1 = pos_df.iloc[0:,1:]
            neg_df1 = neg_df.iloc[0:,1:]
        else:
            pos_df1 = pos_df.iloc[0:,0:]
            neg_df1 = neg_df.iloc[0:,1:]
        
        if  feature_name == 'BPBaac':
            num = len(feature)-236
            neg_target = np.zeros((num))
            pos_target = np.ones((236))
            neg_target_series = pd.Series(neg_target)
            pos_target_series = pd.Series(pos_target)
        else:
            feature = pd.concat([pos_df1, neg_df1])
            neg_target = np.zeros((len(neg_df1)))
            pos_target = np.ones((len(pos_df1)))
            neg_target_series = pd.Series(neg_target)
            pos_target_series = pd.Series(pos_target)
    

    
        target = pd.concat([pos_target_series, neg_target_series], ignore_index=True)
        
        if feature_name == 'CTriad':
    
            feature_ = np.array([eval(row) for row in feature['CTriad']])
            target_ = target.values
        else:
            feature_ = feature.astype("float").values
            target_ = target.values
        
        class MyOptimitzer:
            def __init__(self, classifier_name: str, classifier_class: ClassifierMixin, classifier_param_dict: dict) -> None:
                self.classifier_name = classifier_name
                self.classifier_class = classifier_class
                self.classifier_param_dict = classifier_param_dict

                self.grid_search: BaseSearchCV = None
                self.train_best_predicted_pair = None
                self.train_best_5C_predicted_pair = None
                self.best_predicted_pair = None
                self.best_5C_predicted_pair = None
                self.start_to_train_time = datetime.now()
                self.end_of_train_time = None
                pass

            def find_best(
                self,
                X: np.ndarray,
                y: np.ndarray,
                validation: tuple,
                search_method: typing.Literal["GridSearchCV", "BayesSearchCV"],
                n_jobs: int = 31
            ):

                

                if search_method == "GridSearchCV":
                    self.grid_search = GridSearchCV(
                        self.classifier_class(),
                        param_grid=self.classifier_param_dict,
                        cv=StratifiedKFold(
                            n_splits=5,
                            shuffle=True,
                            random_state=42
                        ),
                        scoring='roc_auc',
                        n_jobs=n_jobs,
                        refit=True
                    )
                elif search_method == "BayesSearchCV":
                    self.grid_search = BayesSearchCV(
                        self.classifier_class(),
                        search_spaces=self.classifier_param_dict,
                        cv=StratifiedKFold(
                            n_splits=5,
                            shuffle=True,
                            random_state=42
                        ),
                        scoring='roc_auc',
                        n_jobs=n_jobs,
                        n_points=n_jobs,
                        n_iter=5,
                        refit=True
                    )
                else:
                    raise ValueError(
                        'search_method: typing.Literal["GridSearchCV", "BayesSearchCV"]'
                    )
                y_origin = y
                if self.classifier_name == "LabelPropagation":
                    y = y.copy()
                    y[
                        np.random.choice(
                            a=np.arange(X.shape[0]),
                            size=max(int(X.shape[0] * 0.25), 1)
                        )
                    ] = -1
                full_X = np.concatenate([
                    X, validation[0]
                ])
                full_y = np.concatenate([
                    y_origin, validation[1]
                ])

                self.grid_search.fit(full_X, full_y)
                self.best_predicted_pair = [
                    np.nan_to_num(self.grid_search.predict_proba(
                        X=validation[0]
                    ), nan=0.0),
                    validation[1]
                ]
                self.train_best_predicted_pair = [
                    np.nan_to_num(self.grid_search.predict_proba(
                        X=X
                    ), nan=0.0),
                    y
                ]

                # 5倍交叉验证
                
                # 跑模型
                self.best_5C_predicted_pair = []
                self.train_best_5C_predicted_pair = []
                for Kfold_id, (train_id, test_id) in enumerate(
                    StratifiedKFold(
                        n_splits=5,
                        shuffle=True,
                        random_state=42
                    ).split(full_X, full_y)
                ):
                    

                    # 定义模型并加载参数
                    fiveC_model = self.classifier_class(
                        **self.grid_search.best_params_,
                    )
                    y_to_train = full_y[train_id].copy()
                    if self.classifier_name == "LabelPropagation":
                        y_to_train[
                            np.random.choice(
                                a=np.arange(y_to_train.shape[0]),
                                size=max(int(y_to_train.shape[0] * 0.25), 1)
                            )
                        ] = -1

                    
                    fiveC_model.fit(
                        full_X[train_id],
                        y_to_train
                    )

                    # 预测并记录
                    self.best_5C_predicted_pair.append([
                        np.nan_to_num(fiveC_model.predict_proba(
                            X=full_X[test_id]
                        ), nan=0.0),
                        full_y[test_id]
                    ])
                    self.train_best_5C_predicted_pair.append([
                        np.nan_to_num(fiveC_model.predict_proba(
                            X=full_X[train_id]
                        ), nan=0.0),
                        y_to_train
                    ])

                return self

            def get_summary(self, path_to_dir: str = None):
                os.makedirs(path_to_dir, exist_ok=True)
                model_path = "-"
                

                model_path = f"{path_to_dir}/{self.classifier_name}.pkl"
                if path_to_dir is not None:
                    with open(model_path, "bw+") as f:
                        pickle.dump(
                            self.grid_search, f
                        )
                    

                

                training_testing_performance = get_evaluation(
                    label=self.best_predicted_pair[1],
                    pred=self.best_predicted_pair[0][:, 1],
                )

                # 计算5C中的平均表现
                FiveFold_result = {}
                for keys in training_testing_performance.keys():
                    value_list = []
                    for item in self.best_5C_predicted_pair:

                        item_performance = get_evaluation(
                            label=item[1],
                            pred=item[0][:, 1],
                        )
                        value_list.append(item_performance[keys])

                    if keys == "pro_cutoff":
                        FiveFold_result[keys] = value_list
                    else:
                        FiveFold_result[keys] = sum(value_list) / len(value_list)

                self.end_of_train_time = datetime.now()

                return pd.Series({
                                "Classifier_Name": self.classifier_name,
                                "Optimitied_Param": dict(self.grid_search.best_params_),
                                "Model_Path": model_path
                            } | FiveFold_result
                                )

        find_space = [
                {
                 "name": "SVC",
                 "class": SVC,
                 "param": {
                     'C': [0.1, 1, 10, 100],
                     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                     'gamma': ['scale', 'auto'],
                     'degree': [2, 3, 4],
                     'coef0': [0, 1, 2],
                     "probability": [True, ]
                    }
                },
                {
                    "name": "GradientBoostingClassifier",
                    "class": GradientBoostingClassifier,
                    "param": {
                        'learning_rate': [0.01, 0.1, 1],
                        'n_estimators': [10, 50, 100, 500],
                        'max_depth': [3, 5, 7],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'max_features': ['sqrt', 'log2', None]
                    }
                },
                {
                    "name": "GaussianNB",
                    "class": GaussianNB,
                    "param": {
                        "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
                    }
                },
                {
                    "name": "GaussianProcessClassifier",
                    "class": GaussianProcessClassifier,
                    "param": {
                        'max_iter_predict': [10, 50, 100],
                        'warm_start': [True, False],
                        "n_restarts_optimizer": [0, 1, 2],
                    }
                },
                {
                    "name": "KNeighborsClassifier",
                    "class": KNeighborsClassifier,
                    "param": {
                        "n_neighbors": [3, 5, 7, 9],
                        "weights": ["uniform", "distance"],
                        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                        "leaf_size": [10, 30, 50]
                    }
                },
                {
                    "name": "RandomForestClassifier",
                    "class": RandomForestClassifier,
                    "param": {
                        'n_estimators': [10, 20, 50, 100],
                        'criterion': ['gini', 'entropy'],
                        'max_depth': [None, 5, 10],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'max_features': ['sqrt', 'log2', None]
                    }
                },
                {
                    "name": "XGBClassifier",
                    "class": XGBClassifier,
                    "param": {
                        "booster": ['gbtree', 'gblinear', 'dart'],
                        "learning_rate": [0.01, 0.1, 0.5, 1],
                        "reg_lambda": [0, 1, 10],
                        "n_estimators": [50, 100, 200]
                    }
                },
            ]
        model_path_to_save = f'/mnt/md0/Public/T3_T4/model/bac_last_model/{bac_name}/{feature_name}'
        os.makedirs(model_path_to_save, exist_ok=True)

        result_list = []
        for model_index in tqdm(range(len(find_space))):
            fivecross_result = pd.concat([
                MyOptimitzer(
                    find_space[model_index]["name"],
                    find_space[model_index]["class"],
                    find_space[model_index]["param"],
                ).find_best(
                    X=feature_[train_id],
                    y=target_[train_id],
                    search_method = "BayesSearchCV",
                    validation=(feature_[test_id], target_[test_id])
                ).get_summary(
                    path_to_dir=f"{model_path_to_save}/"
                )
                for Kfold_id, (train_id, test_id) in enumerate(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(feature_, target_))
            ], axis=1).T

            print(fivecross_result)

            fivecross_result.loc[:, ["Classifier_Name", "Optimitied_Param", "Model_Path"]].to_csv(
                f"{model_path_to_save}/{find_space[model_index]['name']}_Param.csv"
            )
            fivecross_result_splited = fivecross_result.loc[:, [
                "accuracy", "precision", "f1_score", "mmc", "rocAUC", "specificity", "sensitivity", "pro_cutoff","prAUC"]]
            fivecross_result_splited.to_csv(
                f"{model_path_to_save}/{find_space[model_index]['name']}_5Fold.csv"
            )

            series = fivecross_result_splited.sum(axis=0)
            series.name = find_space[model_index]["name"]
            result_list.append(series)

        pd.concat(
            result_list, axis=1,
        ).T.to_csv(
            f"{model_path_to_save}/5fold_results.csv",
            index=True
        )
            
        