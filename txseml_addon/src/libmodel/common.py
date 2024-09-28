'''
Author: George Zhao
Date: 2022-02-11 19:55:46
LastEditors: George Zhao
LastEditTime: 2022-03-06 15:37:02
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''


class Model_Selection:
    def __init__(self, param_o, cv, default_param) -> None:
        self.param_o = param_o
        self.cv = cv
        self.default_param = default_param
        pass

    def best_fit(self, X, y, verbose=-1, n_jobs=1):
        return self

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return

    def find_parm(self, X, y, n_jobs, verbose):
        return self


class Model_Final:
    def __init__(self, cv, desc='unDefine'):
        self.cv = cv
        self.desc = desc

    def tranmodel(
        self,
        f,
        l,
    ):
        return self

    def predict(self, f):
        return

    def save_to_file(self, path_to_dir):
        return self

    def load_model(self, path_to_dir, ):
        return self

    def clean_model(self):
        return self
