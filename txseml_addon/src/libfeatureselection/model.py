import os
import typing
from datetime import datetime

from sklearn.base import ClassifierMixin

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection._search import BaseSearchCV
from skopt import BayesSearchCV

import numpy as np
import pandas as pd
import pickle
import gzip

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


from sklearn.metrics import roc_curve, confusion_matrix, precision_score, accuracy_score, f1_score, matthews_corrcoef, auc


def get_evaluation(label: list, pred: list, pro_cutoff: float = None):
    pred = np.nan_to_num(
        pred, copy=True, nan=0.0
    )
    fpr, tpr, thresholds = roc_curve(label, pred)
    if pro_cutoff is None:
        best_one_optimal_idx = np.argmax(tpr - fpr)
        pro_cutoff = thresholds[best_one_optimal_idx]
    pred_l = [1 if i >= pro_cutoff else 0 for i in pred]
    confusion_matrix_1d = confusion_matrix(label, pred_l).ravel()
    confusion_dict = {N: n for N, n in zip(['tn', 'fp', 'fn', 'tp'], list(
        confusion_matrix_1d * 2 / np.sum(confusion_matrix_1d)))}
    evaluation = {
        "accuracy": accuracy_score(label, pred_l),
        "precision": precision_score(label, pred_l),
        "f1_score": f1_score(label, pred_l),
        "mmc": matthews_corrcoef(label, pred_l),
        "rocAUC": auc(fpr, tpr),
        "specificity": confusion_dict['tn'] / (confusion_dict['tn'] + confusion_dict['fp']),
        "sensitivity": confusion_dict['tp'] / (confusion_dict['tp'] + confusion_dict['fn']),
        # "confusion_matrix": confusion_dict,
        # "_roc_Data": {'fpr': list(fpr), 'tpr': list(tpr)},
        'pro_cutoff': pro_cutoff
    }
    return evaluation


def plot_roc_curve(target, pred, path_to_: str):
    fpr, tpr, thresholds = roc_curve(target, pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(19.2 / 4, 10.8 / 4))
    plt.axis('square')
    plt.plot(
        fpr, tpr, color='red', lw=2,
        label='ROC curve (area = %0.2f)' % roc_auc
    )
    plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) curve')
    plt.legend(loc="lower right")

    plt.savefig(f"{path_to_}", transparent=True)
    plt.clf()


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
        self.scaler = None
        self.start_to_train_time = datetime.now()
        self.end_of_train_time = None
        pass

    def find_best(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation: tuple,
        search_method: typing.Literal["GridSearchCV", "BayesSearchCV"],
        n_jobs: int = 1
    ):

        self.scaler = MinMaxScaler()
        self.scaler.fit(X)

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
        self.grid_search.fit(self.scaler.transform(X), y)
        self.best_predicted_pair = [
            np.nan_to_num(self.grid_search.predict_proba(
                X=self.scaler.transform(validation[0])
            ), nan=0.0),
            validation[1]
        ]
        self.train_best_predicted_pair = [
            np.nan_to_num(self.grid_search.predict_proba(
                X=self.scaler.transform(X)
            ), nan=0.0),
            y
        ]

        # 5倍交叉验证
        # 合并数据
        full_X = np.concatenate([
            X, validation[0]
        ])
        full_y = np.concatenate([
            y_origin, validation[1]
        ])

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
            # 放缩器
            _scaler = MinMaxScaler()
            _scaler.fit(full_X[train_id])

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

            # 放缩，并fit一下
            fiveC_model.fit(
                _scaler.transform(full_X[train_id]),
                y_to_train
            )

            # 预测并记录
            self.best_5C_predicted_pair.append([
                np.nan_to_num(fiveC_model.predict_proba(
                    X=_scaler.transform(full_X[test_id])
                ), nan=0.0),
                full_y[test_id]
            ])
            self.train_best_5C_predicted_pair.append([
                np.nan_to_num(fiveC_model.predict_proba(
                    X=_scaler.transform(full_X[train_id])
                ), nan=0.0),
                y_to_train
            ])

        return self

    def get_summary(self, path_to_dir: str = None):
        os.makedirs(path_to_dir, exist_ok=True)
        model_path = "-"
        if "SAVE_MODEL" in os.environ and os.environ['SAVE_MODEL'] == "1":

            model_path = f"{path_to_dir}/{self.classifier_name}.pkl"
            if path_to_dir is not None:
                with gzip.open(model_path, "wb") as f:
                    pickle.dump(
                        self.grid_search, f
                    )
            scaler_path = f"{path_to_dir}/scaler.pkl"
            if path_to_dir is not None:
                with gzip.open(scaler_path, "wb") as f:
                    pickle.dump(
                        self.scaler, f
                    )

        model_score_path = f"{path_to_dir}/{self.classifier_name}_score.pkl"
        if path_to_dir is not None:
            with gzip.open(model_score_path, "wb") as f:
                pickle.dump(
                    {
                        "best_predicted_pair": self.best_predicted_pair,
                        "best_5C_predicted_pair": self.best_5C_predicted_pair,
                    }, f
                )
            with gzip.open(model_score_path + ".train", "wb") as f:
                pickle.dump(
                    {
                        "best_predicted_pair": self.train_best_predicted_pair,
                        "best_5C_predicted_pair": self.train_best_5C_predicted_pair,
                    }, f
                )
        else:
            model_score_path = "-"

        plot_roc_curve(
            target=self.best_predicted_pair[1],
            pred=self.best_predicted_pair[0][:, 1],
            path_to_=f"{path_to_dir}/{self.classifier_name}.pdf"
        )

        model_information = {
            "Classifier_Name": self.classifier_name,
            "Optimitied_Param": dict(self.grid_search.best_params_),
            "Score": model_score_path,
            "Model_Path": model_path,
            "TimeToStartFit": self.start_to_train_time.strftime("%Y-%m-%d %H:%M:%S")
        }

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
        model_information["TimeOfSummary"] = self.end_of_train_time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        model_information["TimeSpend"] = str(
            self.end_of_train_time - self.start_to_train_time
        )

        return model_information, training_testing_performance, FiveFold_result
