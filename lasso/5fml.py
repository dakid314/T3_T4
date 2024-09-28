import os
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd

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
from sklearn.metrics import roc_curve, confusion_matrix, precision_score, accuracy_score, f1_score, matthews_corrcoef, auc

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_curve, confusion_matrix, precision_score, accuracy_score, f1_score, matthews_corrcoef, auc,precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
def get_evaluation(label: list, pred: list, pro_cutoff: float = None):
    fpr, tpr, thresholds = roc_curve(label, pred)
    if pro_cutoff is None:
        best_one_optimal_idx = np.argmax(tpr - fpr)
        pro_cutoff = thresholds[best_one_optimal_idx]
    pred_l = [1 if i >= pro_cutoff else 0 for i in pred]
    #后面新增的计算prAUC
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

class MyOptimitzer:
        def __init__(self, classifier_name: str, classifier_class: ClassifierMixin, classifier_param_dict: dict) -> None:
            self.classifier_name = classifier_name
            self.classifier_class = classifier_class
            self.classifier_param_dict = classifier_param_dict

            self.grid_search: GridSearchCV = None
            self.best_predicted_pair = None
            pass

        def find_best(self, X, y, validation: tuple):
            self.grid_search = BayesSearchCV(
                self.classifier_class(),
                search_spaces=self.classifier_param_dict,
                cv=RepeatedStratifiedKFold(
                    n_splits=5, #原来是5
                    n_repeats=2,
                    random_state=42
                ),
                scoring='roc_auc',
                # n_jobs=-3, #控制线程 ：正数表示使用多少线程；负数表示在系统总线程上减去相应线程来使用
                n_jobs=-4,
                refit=True
            )
            self.grid_search.fit(X, y)
            self.best_predicted_pair = [
                self.grid_search.predict_proba(
                    X=validation[0]
                ),
                validation[1]
            ]
            return self

        def get_summary(self, path_to_dir: str = None):
            os.makedirs(path_to_dir, exist_ok=True)
            model_path = f"{path_to_dir}/{self.classifier_name}.pkl"
            if path_to_dir is not None:
                with open(model_path, "bw+") as f:
                    pickle.dump(
                        self.grid_search, f
                    )
            else:
                model_path = "-"
            plot_roc_curve(
                target=self.best_predicted_pair[1],
                pred=self.best_predicted_pair[0][:, 1],
                path_to_=f"{path_to_dir}/{self.classifier_name}.pdf"
            )
            return pd.Series({
                "Classifier_Name": self.classifier_name,
                "Optimitied_Param": self.grid_search.best_params_,
                "Model_Path": model_path
            } | get_evaluation(
                label=self.best_predicted_pair[1],
                pred=self.best_predicted_pair[0][:, 1],
            ))
find_space = [
    #     {
    #     "name": "LogisticRegression",
    #     "class": LogisticRegression,
    #     "param": [ {
    #         "penalty": ['l1', ],
    #         "C": [0.001, 0.01, 0.1, 1, 10, 100],
    #         "solver": ['liblinear', 'saga'],
    #         'class_weight': [None, 'balanced'],
    #         'max_iter': [100, 500, 1000]
    #     }, {
    #         "penalty": ['l2', ],
    #         "C": [0.001, 0.01, 0.1, 1, 10, 100],
    #         "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    #         'class_weight': [None, 'balanced'],
    #         'max_iter': [100, 500, 1000]
    #     }, {
    #         "penalty": ['elasticnet', ],
    #         'l1_ratio': [0.1 * i for i in range(0, 11, 1)],
    #         "C": [0.001, 0.01, 0.1, 1, 10, 100],
    #         "solver": ['saga'],
    #         'class_weight': [None, 'balanced'],
    #         'max_iter': [100, 500, 1000]
    #     }, ]
    # },
    {
        "name": "LogisticRegression",
        "class": LogisticRegression,
        "param": [{
            "penalty": ['l2', ],
            "C": [0.1, 1.0, ],
            "solver": ['lbfgs',],
            'class_weight': ['balanced',],
            'max_iter': [1000, ]
        }, ],
        "Bayes": False
    },
    ]
a = 1
#读取数据
f = pd.read_csv(f'new_LR/{a}/feature.csv')
feature = f.iloc[0:,1:-1]
feature_ = feature.astype("float").values
target = f.loc[0:,'label']
target_ = target.values

model_path_to_save = f'new_LR_test/{a}'
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
            validation=(feature_[test_id], target_[test_id])
        ).get_summary(
            path_to_dir=f"{model_path_to_save}/{Kfold_id}"
        )
        for Kfold_id, (train_id, test_id) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(feature_, target_))
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

    series = fivecross_result_splited.sum(axis=0) / 5
    series.name = find_space[model_index]["name"]
    result_list.append(series)

pd.concat(
                result_list, axis=1,
            ).T.to_csv(
                f"{model_path_to_save}/5fold_results.csv",
                index=True
            )
