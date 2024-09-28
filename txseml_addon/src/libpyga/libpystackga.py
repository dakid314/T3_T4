'''
Author: George Zhao
Date: 2021-11-03 19:00:47
LastEditors: George Zhao
LastEditTime: 2022-02-27 21:14:23
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
# %%
import sys
sys.path.append('../..')
import os

import itertools
import math
import json

import utils
work_Dir = utils.workdir.workdir(os.getcwd(), 3)
logger = utils.glogger.Glogger('libpystackga', os.path.join(
    work_Dir, 'log'), timestamp=False, std_err=False)

import numpy as np
from sklearn.model_selection import cross_val_score
import tqdm
import joblib

# %%


def load_ds(
    path_to_json_list: list,
    index_: int = 0
):
    # Make Sure that All position in the table are point to the same Protein.
    # Feature
    db = None
    feature = list()
    for path_to_json in path_to_json_list:
        with open(path_to_json, 'r', encoding='UTF-8') as f:
            db = json.load(f)
        feature.append(np.array(db[index_]['test']['origin']['pred']))
    # Label
    label = db[index_]['test']['origin']['label']
    return np.array(feature).T, np.array(label)


class stack_ga:
    def __init__(
        self,
        stacker,
        model_pool_size,
        scoring='roc_auc',
        cv=10,
        verbose=-1,
        n_jobs=-1,
        desc=''
    ):

        # stacker is the stack function, Need a build Function, like: lambda _: SVG()
        self.stacker = stacker
        # The Num of the Model we want to select to stack.
        self.model_pool_size = model_pool_size
        # Measure which is the best one.
        self.scoring = scoring
        # n-Fold when Measure which is the best one.
        self.cv = cv
        # Is verbose?
        self.verbose = verbose
        # n_jobs?
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count() + n_jobs
        # Description.
        self.desc = desc

        # To save the random conbination of the models.
        self.step1_msparam_collection = list()

        # To save what conbination of the models is the best one.
        self.best_msparam_ = None
        self.best_score = None
        self.history_best_score = list()

        # To save population_ elements.
        self.population_ = None

        # To save what conbination we have try.
        self.history = list()

    def clean(
        self
    ):
        self.step1_msparam_collection = list()
        self.best_msparam_ = None
        self.best_score = None
        self.history_best_score = list()
        self.history = list()

    def _step1(
        self,
        n_Individual: int = 25,
        per_model_probi: float = 0.5
    ):
        # n_Individual mean that How many Individual you want in the population.
        self.step1_msparam_collection = np.reshape(a=np.random.choice(
            a=[1, 0],
            size=self.model_pool_size * n_Individual,
            replace=True,
            p=[per_model_probi, 1 - per_model_probi]
        ), newshape=(n_Individual, self.model_pool_size))
        return self

    def _step2(
        self,
        X, y,
        n_iter: int = 10,
        p_of_mutation: float = 0.5,
        p_of_recombination: float = 0.5,
        n_Population_scale: int = 2,
        early_stop_wait_iter: int = 5,
        stacker_param={}
    ):
        # You will get n_Population_scale * self.step1_msparam_collection Individual in your Population.
        # It will iter n_iter Time.
        if X.shape[1] != self.model_pool_size:
            raise ValueError(
                f'In Step2: X.shape[1] != self.model_pool_size: {X.shape[1]} != {self.model_pool_size}.')

        def _bar_type1(x): return iter(x)
        def _bar_type2(x): return tqdm.tqdm(x, desc=self.desc)
        if self.verbose > 0:
            bar_type = _bar_type2
        else:
            bar_type = _bar_type1

        def fold_v_function(stacker_f, vector_choise=None):

            # ? Change what?

            if vector_choise is not None and sum(vector_choise) != 0:
                feature_choised = X[
                    :,
                    [
                        e_index
                        for e_index, element in
                        enumerate(vector_choise)
                        if element == 1
                    ]
                ]
            else:
                feature_choised = X
            return np.mean(
                cross_val_score(
                    estimator=stacker_f,
                    X=feature_choised,
                    y=y,
                    scoring=self.scoring,
                    cv=self.cv))
        # Prepare self.population_.
        self.population_ = list(itertools.chain(*[
            [idv
             for _ in range(n_Population_scale)]
            for idv in self.step1_msparam_collection]))
        num_of_population = n_Population_scale *\
            len(self.step1_msparam_collection)

        # Evolution.
        # early_stop_count iter num.
        early_stop_count = 0
        for n_th_iter in range(n_iter):
            logger.logger.info(
                f'This is the {n_th_iter}/{n_iter} iter of Evolution.')
            # mutation
            mutation_rng = np.random.random(
                size=(len(self.population_), self.model_pool_size))
            for inv_index in range(mutation_rng.shape[0]):
                # inv_index: Individual Num.
                for feature_index in range(mutation_rng.shape[1]):
                    # feature_index: Model Index.
                    if mutation_rng[inv_index][feature_index] < p_of_mutation:
                        # to mutation
                        if self.population_[inv_index][feature_index] == 0:
                            self.population_[inv_index][feature_index] = 1
                        else:
                            self.population_[inv_index][feature_index] = 0

            # Recombination.
            # Prepare Coit Pair.
            coit_pair = np.reshape(
                np.random.choice(
                    np.arange(len(self.population_)),
                    size=math.floor(len(self.population_) / 2) * 2,
                    replace=False),
                (-1, 2)
            )

            recombination_rng = np.random.random(
                size=(coit_pair.shape[0], self.model_pool_size))
            for coit_pair_index in range(recombination_rng.shape[0]):
                # For the coit_pair_index-th Pair.
                # Male and Female.
                m, f = coit_pair[coit_pair_index]
                for feature_index in range(recombination_rng.shape[1]):
                    # feature_index: Model Index.
                    if recombination_rng[coit_pair_index, feature_index] < p_of_recombination:
                        # to recombination.
                        self.population_[m][feature_index], self.population_[f][feature_index] = self.population_[
                            f][feature_index], self.population_[m][feature_index]

            # ! To Check.
            # Calculate the Survive Scroes of Individual in the Population.
            population_scores = joblib.Parallel(
                n_jobs=self.n_jobs
                if len(self.population_) >= self.n_jobs else
                len(self.population_)
            )(
                joblib.delayed(fold_v_function)(
                    self.stacker(**stacker_param),
                    model_selection
                ) for model_selection in
                bar_type(self.population_)
            )

            # Rank the Scores of Individual.
            # 1. Get the Best One.
            # 2. Store the History.
            # 3. Perpare the New Population.
            # 4. Early Stop.

            # Perpare clean-Data.
            population_p_s_list_ranked = dict()
            for paramdict, scors in zip(self.population_, population_scores):
                param_pair_name = ';'.join(
                    [
                        f'{k}' for k in paramdict
                    ]
                )
                if param_pair_name not in population_p_s_list_ranked.keys():
                    population_p_s_list_ranked[param_pair_name] = {
                        'dict': paramdict,
                        'scors': []}
                population_p_s_list_ranked[param_pair_name]['scors'].append(
                    scors)
            # Calculate the scors_mean.
            for k in population_p_s_list_ranked:
                population_p_s_list_ranked[k]['scors_mean'] = np.mean(
                    population_p_s_list_ranked[k]['scors'])

            # 1. Get the LOCAL best individual.
            best_individual = population_p_s_list_ranked[
                max(
                    population_p_s_list_ranked,
                    key=lambda k: population_p_s_list_ranked[k]['scors_mean']
                )
            ]

            local_best_msparam_ = best_individual['dict']

            local_best_score = {
                'scores_mean': best_individual['scors_mean'],
                'scores': best_individual['scors']
            }

            # 2. Store the History.
            self.history.append(
                population_p_s_list_ranked
            )
            self.history_best_score.append(local_best_score)

            # 4. Early Stop.
            if len(self.history_best_score) <= 1:
                # 1.1 Store Best Result.
                self.best_msparam_ = local_best_msparam_
                self.best_score = local_best_score
            else:
                if local_best_score['scores_mean'] <= self.history_best_score[-1 * (early_stop_count + 2)]['scores_mean']:
                    logger.logger.info(
                        f'early_stop_count:{early_stop_count} += 1')
                    early_stop_count += 1
                else:
                    logger.logger.info(
                        f'early_stop_count:{early_stop_count} set to 0.')
                    early_stop_count = 0
                    # 1.1 Store Best Result.
                    self.best_msparam_ = local_best_msparam_
                    self.best_score = local_best_score
            if early_stop_count > early_stop_wait_iter:
                break

            # 3. Perpare the New Population.
            # 3.1 Weight of Each One.
            all_scors_mean = np.array([population_p_s_list_ranked[k]
                                       ['scors_mean'] for k in population_p_s_list_ranked.keys()])
            softmax_denominator = np.sum(np.exp(all_scors_mean))
            all_num_of_popula = np.exp(
                all_scors_mean) / softmax_denominator * num_of_population

            # 3.2 Generate the Population.
            self.population_ = list(itertools.chain(*[
                [
                    population_p_s_list_ranked[param_dict_key]['dict']
                    for _ in range(math.floor(n))
                ]
                for param_dict_key, n in zip(population_p_s_list_ranked.keys(), all_num_of_popula)])
            )

            # 3.3 Extend Population to num_of_population.
            self.population_.extend([self.best_msparam_ for _ in range(
                max(1, num_of_population - len(self.population_)))])
            logger.logger.info(
                f'The {n_th_iter}/{n_iter} iter of Evolution Over.')

        return self

    def fit(
        self,
        X, y,
        **step_param
    ):
        step1_param = {
        } if 1 not in step_param['step_param'] else step_param['step_param'][1]
        step2_param = {
        } if 2 not in step_param['step_param'] else step_param['step_param'][2]
        logger.logger.info('Prepare For Step1.')
        self._step1(**step1_param)
        logger.logger.info('Prepare For Step2.')
        self._step2(X, y, **step2_param)
        logger.logger.info('Optimition Ready.')
        return self


def get_stack(
    path_to_json_list: list,
    name: dict,
    index_: int,
    stacker_f,
    step_param_config: dict,
    predict_func=lambda m, X: m.predict_proba(X)[:, 1],
    desc='Undefind'
):

    path_to_json_list_training = [
        os.path.join(p, *[name['traning'], ]) for p in path_to_json_list
    ]
    path_to_json_list_test = [
        os.path.join(p, *[name['test'], ]) for p in path_to_json_list
    ]

    # Fit
    db_f, db_l = load_ds(
        path_to_json_list=path_to_json_list_training,
        index_=index_
    )

    r = stack_ga(
        stacker=stacker_f,
        model_pool_size=len(path_to_json_list),
        verbose=1,
        desc=desc,
    )
    r.fit(
        X=db_f, y=db_l, step_param=step_param_config
    )

    # Get Index
    model_list_index = [
        m for model_index_switch, m in zip(r.best_msparam_, path_to_json_list) if model_index_switch == 1]

    db_choise_f = db_f[:, [
        e_index
        for e_index, element in
        enumerate(r.best_msparam_)
        if element == 1
    ]]

    # # Get PRED DB.
    db_pred_f, db_pred_l = load_ds(
        path_to_json_list=path_to_json_list_test,
        index_=index_
    )
    db_pred_choise_f = db_pred_f[:, [
        e_index
        for e_index, element in
        enumerate(r.best_msparam_)
        if element == 1
    ]]

    # Get Model Classifier
    classifier_param = {}
    if 'stacker_param' in step_param_config[2]:
        classifier_param.update(step_param_config[2]['stacker_param'])

    # # Get Model
    model_final = stacker_f(**classifier_param)
    model_final = model_final.fit(
        db_choise_f, db_l
    )

    # Predict
    result_dict = {
        "test": {
            "origin": {
                f'pred': list(predict_func(model_final, db_pred_choise_f)),
                f'label': list(db_pred_l)},
            "evaluation": {
            }
        },
    }
    return model_final, result_dict, model_list_index, r


if __name__ == '__main__':
    r = stack_ga(None, 10)
    r._step1()
    # r._step2(None, None)

# %%
