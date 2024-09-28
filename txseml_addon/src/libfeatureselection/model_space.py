from sklearn.semi_supervised import LabelPropagation
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

find_space = [
    {
        "name": "XGBClassifier",
        "class": XGBClassifier,
        "param": {
            "booster": ['gbtree', ],
            "learning_rate": [0.1,],
            "n_estimators": [50, ],
            "random_state": [42, ],
            "verbosity": [0, ]
            # "n_jobs": [1, ]
        },
        "Bayes": False
    },
    {
        "name": "LGBMClassifier",
        "class": LGBMClassifier,
        "param": {
            'boosting_type': ['gbdt', ],
            'learning_rate': [0.1,],
            'n_estimators': [50, ],
            'verbose': [-1, ],
            "random_state": [42, ]
        },
        "Bayes": False
    },
    {
        "name": "LabelPropagation",
        "class": LabelPropagation,
        "param": {
            'kernel': ['knn', 'rbf'],
            'gamma': [0.1, 1, 10,],
            'n_neighbors': [1, 3, 5],
            'max_iter': [1000, ],
        },
        "Bayes": False
    },
    {
        "name": "SVC",
        "class": SVC,
        "param": {
            'C': [0.01, 0.1, 1,],
            'kernel': ['linear', 'rbf', ],
            'gamma': ['auto', ],
            "probability": [True, ]
        },
        "Bayes": False
    },
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
    {
        "name": "SGDClassifier",
        "class": SGDClassifier,
        "param": {
            'loss': ['log', ],
            'penalty': ['l2',],
            'learning_rate': ["optimal",]
        },
        "Baye": False
    },
    {
        "name": "DecisionTreeClassifier",
        "class": DecisionTreeClassifier,
        "param": {
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 5, 7, ],
            'max_features': ['sqrt', 'log2',],
            'random_state': [42,],
            'class_weight': ['balanced',]
        },
        "Bayes": False
    },
    {
        "name": "ExtraTreeClassifier",
        "class": ExtraTreeClassifier,
        "param": {
            'criterion': ["gini", "entropy"],
            'max_depth': [3, 5, 7],
            "class_weight": ['balanced',],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', ]
        },
        "Bayes": False
    },
    {
        "name": "ExtraTreesClassifier",
        "class": ExtraTreesClassifier,
        "param": {
            'max_depth': [3, 5, 7],
            "max_features": ["sqrt", "log2", ],
            "criterion": ["gini", "entropy"],
            "bootstrap": [True, ],
            "n_estimators": [25, 50],
            "class_weight": ['balanced_subsample', ],
            "random_state": [42,]
        },
        "Bayes": False
    },
    {
        "name": "GradientBoostingClassifier",
        "class": GradientBoostingClassifier,
        "param": {
            'learning_rate': [0.1,],
            'n_estimators': [25, 50, ],
            'max_depth': [3, 5, 7],
            'max_features': ['sqrt', 'log2', ]
        },
        "Bayes": False
    },
    {
        "name": "GaussianNB",
        "class": GaussianNB,
        "param": {
            "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
        },
        "Bayes": False
    },
    {
        "name": "GaussianProcessClassifier",
        "class": GaussianProcessClassifier,
        "param": {
            'max_iter_predict': [100, ],
            'warm_start': [True, ],
            "random_state": [42, ]
        },
        "Bayes": False
    },
    {
        "name": "MLPClassifier",
        "class": MLPClassifier,
        "param": {
            'hidden_layer_sizes': [(100,), (50, ), (25,)],
            'activation': ['logistic', 'relu'],
            'solver': ['adam',],
            'learning_rate': ['adaptive', ],
            'max_iter': [1000,],
            "early_stopping": [True,],
            "verbose": [False,]
        },
        "Bayes": False
    },
    {
        "name": "KNeighborsClassifier",
        "class": KNeighborsClassifier,
        "param": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", ],
            "leaf_size": [10, 30, 50]
        },
        "Bayes": False
    },
    {
        "name": "RandomForestClassifier",
        "class": RandomForestClassifier,
        "param": {
            'n_estimators': [25, 50,],
            'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2', ],
        },
        "Bayes": False
    },
]
