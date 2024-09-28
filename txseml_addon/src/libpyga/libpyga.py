'''
Author: George Zhao
Date: 2021-07-31 16:12:52
LastEditors: George Zhao
LastEditTime: 2022-02-28 00:33:39
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
# %%
import math
import itertools
import functools
import random
# %%
import tqdm
import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV, cross_val_score

# Base scikit
# %%


def seacond_parm(lgb_param_optim: dict, lgb_param_o: dict, p=True, size_=3):

    # Choise 3(size_) value each key in lgb_param_o and then return.
    # argment: lgb_param_optim and p is unused.

    param_to_opti = dict()
    if p == True:
        for ki in lgb_param_o.keys():
            param_to_opti.update({
                ki: sorted(np.random.choice(
                    lgb_param_o[ki],
                    size=size_ if len(lgb_param_o[ki]) >= size_ else len(
                        lgb_param_o[ki]),
                    replace=False))
            })
    else:
        raise RuntimeError("What happened?")
        for ki in lgb_param_o.keys():
            lgb_param_o[ki] = sorted(lgb_param_o[ki])
        for ki in lgb_param_optim.keys():
            c = lgb_param_o[ki].index(lgb_param_optim[ki])
            param_to_opti.update({
                ki: list(set([
                    lgb_param_o[ki][max(0, c - 1)],
                    lgb_param_optim[ki],
                    lgb_param_o[ki][
                        min(len(lgb_param_o[ki]) - 1, c + 1)
                    ],
                ]))
            })
            pass
    return param_to_opti


class bastion3_ga_cv:
    def __init__(self,
                 estimator,
                 param_o,
                 scoring='roc_auc',
                 refit=True,
                 cv=10,
                 verbose=-1,
                 verbose_bar=0,
                 fit_parament={},
                 n_jobs=-1,
                 desc='Bation3_gaCV'):
        # Model Constructor with default parament: functools.partial(lgb.LGBMClassifier,**self.defalut_para)
        self.estimator = estimator
        # parament to optimite.
        self.param_o = param_o
        # scoring method: 'AUC', 'accuracy'...
        self.scoring = scoring
        # Argment for GridSearchCV.
        self.refit = refit
        # CV: interge 10 or ShuffleSplit(n_splits=10, test_size=0.5, random_state=0)
        self.cv = cv
        # verbose for sklearn api.
        self.fit_parament = fit_parament
        self.verbose = verbose
        self.verbose_bar = verbose_bar if verbose_bar is not None else verbose
        # n_jobs
        self.n_jobs = n_jobs

        self.step1_param_collection = list()

        self.best_param_ = None
        self.best_scorse = None
        self.history_best_score = list()

        self.population_ = None
        self.desc = desc

        self.history = list()

    def _step1(self, X, y, n_Individual: int = 10, param_size_=3, sc_=1, append=False):

        if append == False:
            # append ==True mean that add more into self.step1_param_collection.
            self.step1_param_collection = list()

        for _ in range(n_Individual):

            # Generate n_Individual parament.

            # Generate {keys: [param_size_]}.
            chiosed_param = seacond_parm(
                {}, self.param_o, p=True, size_=param_size_)

            # local optimite parament dict.
            opti_param = dict()

            # Shuffle the key list first.
            key_list = list(chiosed_param.keys())
            random.shuffle(key_list)

            k_l_iter = iter(key_list)
            # 1. Choise sc_ key from k_l_iter.
            for k_l in [
                itertools.islice(k_l_iter, 0, sc_)
                for _ in range(math.ceil(len(chiosed_param.keys()) / sc_))
            ]:
                gs = GridSearchCV(
                    self.estimator(**opti_param),
                    # Get k_l from chiosed_param.
                    {k: chiosed_param[k] for k in k_l},
                    refit=self.refit,
                    cv=self.cv,
                    verbose=self.verbose,
                    n_jobs=self.n_jobs)
                gs.fit(X, y, **self.fit_parament)

                opti_param.update(gs.best_params_)

            # Add opti_param in to collection.
            self.step1_param_collection.append(opti_param)
        return self

    def _step2(self,
               X, y,
               n_iter: int = 10,
               p_of_mutation: float = 0.5,
               p_of_recombination: float = 0.5,
               n_Population_scale: int = 2,
               early_stop_wait_iter: int = 5,
               replace_=True
               ):

        def fold_v_function(e):
            return np.mean(
                cross_val_score(
                    estimator=e,
                    X=X,
                    y=y,
                    scoring=self.scoring,
                    cv=self.cv,
                    fit_params=self.fit_parament
                )
            )

        # num_of_population
        num_of_population = n_Population_scale *\
            len(self.step1_param_collection)

        # How many are the self.param_o.keys and what is.
        feature_keys, num_of_feature = list(
            self.param_o.keys()), len(self.param_o.keys())

        if replace_ == True or self.population_ is None:
            # Generate population from start.
            self.population_ = list(itertools.chain(*[
                [
                    idv
                    for _ in range(n_Population_scale)
                ]
                for idv in self.step1_param_collection
            ]))

        # early_stop_count iter num.
        early_stop_count = 0
        for revolution_iter_index in range(n_iter):
            # Run revolution for n_iter times.

            # mutation
            mutation_rng = np.random.random(
                size=(len(self.population_), num_of_feature))
            for inv_index in range(mutation_rng.shape[0]):
                # individual in population
                for feature_index in range(num_of_feature):
                    # feature index
                    if mutation_rng[inv_index][feature_index] < p_of_mutation:
                        self.population_[inv_index][feature_keys[feature_index]] = np.random.choice(
                            self.param_o[feature_keys[feature_index]], size=1)[0]

            # recombination
            # Which twice want to coit?
            coit_pair = np.reshape(
                np.random.choice(
                    np.arange(len(self.population_)),
                    size=math.floor(len(self.population_) / 2) * 2,
                    replace=False),
                (-1, 2)
            )
            recombination_rng = np.random.random(
                size=(coit_pair.shape[0], num_of_feature))
            for coit_pair_index in range(len(coit_pair)):
                m, f = coit_pair[coit_pair_index]
                # pair index.
                for feature_index in range(num_of_feature):
                    # Feature index.
                    if recombination_rng[coit_pair_index, feature_index] < p_of_recombination:
                        # swap feature.
                        self.population_[m][feature_keys[feature_index]], self.population_[f][feature_keys[feature_index]] = self.population_[
                            f][feature_keys[feature_index]], self.population_[m][feature_keys[feature_index]]

            # display bar
            def bar_type_1(x): return x
            def bar_type_2(x): return tqdm.tqdm(
                x, desc=f'{self.desc}_{revolution_iter_index}')
            bar_type = bar_type_1
            if self.verbose_bar > 0:
                bar_type = bar_type_2

            # Run jobs.
            population_scores = joblib.Parallel(
                n_jobs=self.n_jobs
                if len(self.population_) >= self.n_jobs else
                len(self.population_))(
                joblib.delayed(fold_v_function)(self.estimator(**param)) for param in bar_type(self.population_)
            )

            # Rank the result.

            population_p_s_list_ranked = dict()
            for paramdict, scors in zip(self.population_, population_scores):

                # hashlize the population
                param_pair_name = ';'.join(
                    [
                        f'{k}:{paramdict[k]}' for k in paramdict
                    ]
                )
                if param_pair_name not in population_p_s_list_ranked.keys():
                    population_p_s_list_ranked[param_pair_name] = {
                        'dict': paramdict,
                        'scors': []}

                # Add result scors to population_p_s_list_ranked.
                population_p_s_list_ranked[param_pair_name]['scors'].append(
                    scors)

            # averge the scors in population_p_s_list_ranked.
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
                self.best_param_ = local_best_msparam_
                self.best_scorse = local_best_score['scores_mean']
            else:
                if local_best_score['scores_mean'] <= self.history_best_score[-1 * (early_stop_count + 2)]['scores_mean']:
                    early_stop_count += 1
                else:
                    early_stop_count = 0
                    # 1.1 Store Best Result.
                    self.best_param_ = local_best_msparam_
                    self.best_score = local_best_score['scores_mean']
            if early_stop_count > early_stop_wait_iter:
                break

            all_scors_mean = np.array([population_p_s_list_ranked[k]
                                       ['scors_mean'] for k in population_p_s_list_ranked])
            softmax_denominator = np.sum(np.exp(all_scors_mean))
            all_num_of_popula = np.exp(
                all_scors_mean) / softmax_denominator * num_of_population

            self.population_ = list(itertools.chain(*[
                [
                    population_p_s_list_ranked[param_dict_key]['dict']
                    for _ in range(math.floor(n))
                ]
                for param_dict_key, n in zip(population_p_s_list_ranked.keys(), all_num_of_popula)])
            )

            self.population_.extend([self.best_param_ for _ in range(
                max(1, num_of_population - len(self.population_)))])
        return self

    def fit(self, X, y, **step_param):
        step1_param = {
        } if 1 not in step_param['step_param'] else step_param['step_param'][1]
        step2_param = {
        } if 2 not in step_param['step_param'] else step_param['step_param'][2]
        self._step1(X, y, **step1_param)
        self._step2(X, y, **step2_param)
        return self

    pass


if __name__ == "__main__":
    # Test
    pass

# %%
