'''
Author: George Zhao
Date: 2022-03-18 18:14:35
LastEditors: George Zhao
LastEditTime: 2022-06-26 15:54:08
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import os
import sys
import pickle
import json
import math
import functools

import utils
import libpybiofeature

from . import model_optimite
from . import common
import utils
import libpybiofeature

import tqdm
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

n_jobs = 16
allow_load_from_disk = False
submodel_choise = None
submodel_desc = None
only_tt_model = False

epochs = 100
batch_size = 4
learning_rate = 0.005
momentum = 0.9
early_stop_iter = 10

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # tf.config.experimental.set_virtual_device_configuration(
    #     physical_devices[0],
    #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3400)]
    # )


def get_cnn_model(input_dim: int, word_dim: int = 5):
    # 改过参数其实，960->480; 650->128
    input1 = tf.keras.layers.Input(
        shape=(input_dim, ), name='Input_Layer')

    reshape_layer = tf.keras.layers.Reshape(
        (-1, word_dim, 1), name='reshape_layer')(input1)

    if input_dim % word_dim != 0:
        raise ValueError(
            f"input_dim % word_dim != 0: {input_dim} % {word_dim}")
    kernel_size = word_dim
    if input_dim / word_dim < kernel_size:
        kernel_size = (int(input_dim / word_dim), word_dim)

    conv1 = tf.keras.layers.Conv2D(
        120, kernel_size, padding='same')(reshape_layer)

    pool1 = tf.keras.layers.MaxPooling2D(
        pool_size=kernel_size)(conv1)

    flatten_layer = tf.keras.layers.Flatten()(pool1)

    bnl_layer = tf.keras.layers.BatchNormalization()(flatten_layer)

    dropout2 = tf.keras.layers.Dropout(0.6)(bnl_layer)

    dense1 = tf.keras.layers.Dense(
        128,
        activation=tf.keras.activations.elu,
        # kernel_regularizer=tf.keras.regularizers.l1(0.01),
        activity_regularizer=tf.keras.regularizers.l2(0.01)
    )(dropout2)

    output_layer = tf.keras.layers.Dense(
        1,
        activation=tf.keras.activations.sigmoid
    )(dense1)

    model = tf.keras.models.Model(
        inputs=input1, outputs=output_layer, name='CNN_T4SE_CNN')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=[
            tf.keras.metrics.binary_accuracy,
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    return model


svm_param_o = {
    "gamma": [10**i for i in range(-6, 6 + 1)],
    "C": [10**i for i in range(-6, 6 + 1)],
}

svm_optim: model_optimite.svm_optim = functools.partial(
    model_optimite.svm_optim,
    default_param={
        'verbose': False,
        'probability': True,
        'kernel': 'rbf',
    }
)


class CNNT4SE_Model(common.Model_Final):

    def __init__(self, cv, desc):
        super().__init__(cv, desc=desc)
        self.model_list = None

        self.feature_dividend_list = None
        self.model_selectsvm_list = None
        pass

    def tranmodel(self, f, l, feature_dividend_list, model_selectsvm_list):
        super().tranmodel(f, l)
        self.feature_dividend_list = feature_dividend_list
        self.model_selectsvm_list = model_selectsvm_list

        self.model_list = list()
        for feature_group_index in tqdm.tqdm(range(
            1,
            len(self.feature_dividend_list)
        )):
            self.model_list.append(
                self._tran_per_feature(
                    f.iloc[
                        :,
                        self.feature_dividend_list[feature_group_index - 1]:
                        self.feature_dividend_list[feature_group_index]
                    ],
                    l,
                    model_svm=self.model_selectsvm_list[feature_group_index - 1]
                )
            )
        return self

    def predict(self, f):
        super().predict(f)
        # Because of 2 styles of the Results: cnn[[...],], cnn-svm[[...],[...],], I use np.hstack.
        if submodel_choise is None:
            result = np.hstack([
                self._pred_per_feature(
                    f.iloc[
                        :,
                        self.feature_dividend_list[feature_group_index - 1]:
                        self.feature_dividend_list[feature_group_index]
                    ],
                    self.model_list[feature_group_index - 1],
                )
                for feature_group_index in range(
                    1,
                    len(self.feature_dividend_list)
                )
            ])
            result = result[:, [0, 3, 4]]
            return result.sum(axis=1) / result.shape[1]
        # With submodel_choised
        if isinstance(submodel_choise, int) == False or submodel_choise < 0 or submodel_choise >= 11:
            raise ValueError(f"Wrong submodel_choise: {submodel_choise}")
        if submodel_choise < 9:
            feature_group_index = submodel_choise
        else:
            feature_group_index = submodel_choise - 2
        feature_group_index += 1
        return self._pred_per_feature(
            f.iloc[
                :,
                self.feature_dividend_list[feature_group_index - 1]:
                self.feature_dividend_list[feature_group_index]
            ],
            self.model_list[feature_group_index - 1],
            cut_off=None
        ).T[0 if submodel_choise < 9 else 1]

    def _tran_per_feature(
        self, f, l, model_svm: bool
    ):
        result = list()

        fix_f = self.incrense_col(f)
        cnn_model = get_cnn_model(fix_f.shape[1], 5)
        cnn_model.fit(
            tf.constant(fix_f),
            tf.constant(l),
            verbose=0,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='auc', min_delta=0, patience=10, verbose=0,
                    mode='auto', baseline=None, restore_best_weights=True
                ),
            ]
        )
        result.append(cnn_model)

        if model_svm == True:

            result.append(
                svm_optim(
                    param_o=svm_param_o, cv=self.cv
                ).best_fit(
                    f, l, verbose=-1, n_jobs=n_jobs
                )
            )

        return result

    def _pred_per_feature(
        self, f, model_result_list, cut_off=0.5
    ):
        # CNN
        result = [model_result_list[0].predict(
            self.incrense_col(f)).reshape(-1,), ]
        if len(model_result_list) == 2:
            result.append(model_result_list[1].predict_proba(f))
        result = np.stack(
            result
        ).T
        if cut_off is None:
            return result
        return result >= cut_off

    def save_to_file(self, path_to_dir, firstname=None):
        super().save_to_file(path_to_dir)

        for ith, models in enumerate(self.model_list):
            models[0].save(
                self.filename_code(path_to_dir, firstname, ith)
            )
        return self

    def load_model(self, path_to_dir, firstname=None):
        super().load_model(path_to_dir)
        for ith in range(len(self.model_list)):
            # Structure of self.model_list:
            # [ [cnn,], [cnn,], [cnn,svm], ]
            self.model_list[ith][0] = tf.keras.models.load_model(
                self.filename_code(path_to_dir, firstname, ith)
            )
        return self

    def clean_model(self):
        super().clean_model()
        for ith in range(len(self.model_list)):
            self.model_list[ith][0] = None
        return self

    def filename_code(self, path_to_dir, firstname, ith):
        path_to_dir = os.path.splitext(path_to_dir)[0]
        if firstname is not None and ith is not None:
            if os.path.exists(os.path.join(path_to_dir, *[self.desc, firstname])) == False:
                os.path.join(path_to_dir, *[self.desc, firstname])
            return os.path.join(path_to_dir, *[self.desc, firstname, f'{ith}.h5'])
        else:
            raise RuntimeError(
                f"def filename_code(self, {path_to_dir}, {firstname}, {ith})")

    def incrense_col(self, f, n=5):
        if f.shape[1] % n == 0:
            return f
        if type(f) is pd.DataFrame:
            f = f.values
        return np.hstack([f, np.zeros([f.shape[0], n - f.shape[1] % n])])


def Five_Cross_Get_model(
    feature_dividend_list,
    model_selectsvm_list,
    model_construct_funtion,
    t_5C: list,
    v_f,
    v_l,
    path_to_model_pickle: str,
    path_to_model_result: str,
    size_of_data: str,
):
    model_set = list()
    model_result_set = list()
    loaded_from_disk = False
    if allow_load_from_disk == True:
        if os.path.exists(path_to_model_pickle) == True:
            with open(path_to_model_pickle, 'br') as f:
                model_set = pickle.load(f)
            loaded_from_disk = True
    submodel_desc_ = submodel_desc
    if submodel_choise is not None:
        if submodel_desc_ is None:
            submodel_desc_ = submodel_choise
        path_tmp = os.path.split(path_to_model_result)
        path_to_model_result = os.path.join(
            path_tmp[0], submodel_desc_ + '_' + path_tmp[1])
    for i in range(len(t_5C)):
        train_fl, test_fl = t_5C[i]

        model: CNNT4SE_Model = None
        if loaded_from_disk == True:
            model = model_set[i]
            model.load_model(
                path_to_dir=path_to_model_pickle,
                firstname=f'{i}'
            )
        else:
            model: CNNT4SE_Model = model_construct_funtion()
            model.tranmodel(
                train_fl[0], train_fl[1], feature_dividend_list=feature_dividend_list, model_selectsvm_list=model_selectsvm_list
            )
            model_set.append(model)

        model_result_set.append({
            "training": {
                "origin": {
                    f'pred': list(model.predict(
                        train_fl[0]
                    )),
                    f'label': list(train_fl[1])},
                "evaluation": {
                }, "option": {
                }
            },
            "testing": {
                "origin": {
                    f'pred': list(model.predict(
                        test_fl[0]
                    )),
                    f'label': list(test_fl[1])},
                "evaluation": {
                }, "option": {
                }
            },
            "validated": {
                "origin": {
                    f'pred': list(model.predict(
                        v_f
                    )),
                    f'label': list(v_l)},
                "evaluation": {
                }, "option": {
                }
            },
            "detail": {
                "model": 'CNNT4SE',
                'desc': model.desc,
                'iteration': i,
                "size_of_data": size_of_data,
            }
        })
        if loaded_from_disk == True:
            model.clean_model()
        else:
            model.save_to_file(
                path_to_dir=path_to_model_pickle,
                firstname=f'{i}'
            ).clean_model()
        if submodel_choise is not None:
            model_result_set[-1]['detail']['model'] += f'_{submodel_desc_}'

    model_result_set = utils.ds_preprocess.Five_Cross_Evaluation(
        model_result_set,
        pro_cutoff=0.5,
        mode='loo' if size_of_data == 'small' else None
    )

    if os.path.exists(os.path.split(path_to_model_pickle)[0]) == False:
        os.makedirs(os.path.split(path_to_model_pickle)[0])
    if os.path.exists(os.path.split(path_to_model_result)[0]) == False:
        os.makedirs(os.path.split(path_to_model_result)[0])

    if loaded_from_disk == False:
        with open(path_to_model_pickle, 'bw+') as f:
            pickle.dump(model_set, f)

    with open(path_to_model_result, 'w+', encoding='UTF-8') as f:
        json.dump(model_result_set, f, cls=utils.ds_preprocess.MyEncoder)

    return model_set, model_result_set


def research(path_dict: dict):
    seq_id_dict = None
    with open(path_dict['seq_id'], 'r', encoding='UTF-8') as f:
        seq_id_dict = json.load(f)

    possum_index_dict = None
    with open(path_dict['possum']['index'], 'r', encoding='UTF-8') as f:
        possum_index_dict = json.load(f)

    possum_t_p_f, possum_t_n_f, possum_v_p_f, possum_v_n_f = libpybiofeature.pssmcode.get_all_task_feature(
        possum_index_dict=possum_index_dict,
        path_to_json_seq_id=path_dict['seq_id'],
        feature_name_list=['smoothed_pssm', ],
        path_to_fasta_pattern=path_dict['possum']['fasta_pattern'],
        path_to_with_pattern=path_dict['possum']['pssm_fdb_pattern']
    )

    pssm_t_p_f, pssm_t_n_f, pssm_v_p_f, pssm_v_n_f = libpybiofeature.pssmcode.get_all_task_pssmcode(
        possum_index_dict=possum_index_dict,
        seq_id_dict=seq_id_dict,
        path_to_fasta_with_pattern=path_dict['possum']['fasta_pattern'],
        path_to_with_pattern=path_dict['possum']['pssm_rdb_pattern'],
        length=1000,
        cter=path_dict['model']['cter']
    )

    aac_t_p_f = libpybiofeature.featurebuilder.build_acc_feature(
        path_to_fasta=path_dict['fasta']['t']['p'],
        seq_id_list=seq_id_dict['t']['p'],
        desc='t_p'
    )

    aac_t_n_f = libpybiofeature.featurebuilder.build_acc_feature(
        path_to_fasta=path_dict['fasta']['t']['n'],
        seq_id_list=seq_id_dict['t']['n'],
        desc='t_n'
    )

    aac_v_p_f = libpybiofeature.featurebuilder.build_acc_feature(
        path_to_fasta=path_dict['fasta']['v']['p'],
        seq_id_list=seq_id_dict['v']['p'],
        desc='v_p'
    )

    aac_v_n_f = libpybiofeature.featurebuilder.build_acc_feature(
        path_to_fasta=path_dict['fasta']['v']['n'],
        seq_id_list=seq_id_dict['v']['n'],
        desc='v_n'
    )

    CTDC_t_p_f = libpybiofeature.featurebuilder.build_CTDC_feature(
        path_to_fasta=path_dict['fasta']['t']['p'],
        seq_id_list=seq_id_dict['t']['p'],
        desc='t_p'
    )

    CTDC_t_n_f = libpybiofeature.featurebuilder.build_CTDC_feature(
        path_to_fasta=path_dict['fasta']['t']['n'],
        seq_id_list=seq_id_dict['t']['n'],
        desc='t_n'
    )

    CTDC_v_p_f = libpybiofeature.featurebuilder.build_CTDC_feature(
        path_to_fasta=path_dict['fasta']['v']['p'],
        seq_id_list=seq_id_dict['v']['p'],
        desc='v_p'
    )

    CTDC_v_n_f = libpybiofeature.featurebuilder.build_CTDC_feature(
        path_to_fasta=path_dict['fasta']['v']['n'],
        seq_id_list=seq_id_dict['v']['n'],
        desc='v_n'
    )

    CTDT_t_p_f = libpybiofeature.featurebuilder.build_CTDT_feature(
        path_to_fasta=path_dict['fasta']['t']['p'],
        seq_id_list=seq_id_dict['t']['p'],
        desc='t_p'
    )

    CTDT_t_n_f = libpybiofeature.featurebuilder.build_CTDT_feature(
        path_to_fasta=path_dict['fasta']['t']['n'],
        seq_id_list=seq_id_dict['t']['n'],
        desc='t_n'
    )

    CTDT_v_p_f = libpybiofeature.featurebuilder.build_CTDT_feature(
        path_to_fasta=path_dict['fasta']['v']['p'],
        seq_id_list=seq_id_dict['v']['p'],
        desc='v_p'
    )

    CTDT_v_n_f = libpybiofeature.featurebuilder.build_CTDT_feature(
        path_to_fasta=path_dict['fasta']['v']['n'],
        seq_id_list=seq_id_dict['v']['n'],
        desc='v_n'
    )

    CTDD_t_p_f = libpybiofeature.featurebuilder.build_CTDD_feature(
        path_to_fasta=path_dict['fasta']['t']['p'],
        seq_id_list=seq_id_dict['t']['p'],
        desc='t_p'
    )

    CTDD_t_n_f = libpybiofeature.featurebuilder.build_CTDD_feature(
        path_to_fasta=path_dict['fasta']['t']['n'],
        seq_id_list=seq_id_dict['t']['n'],
        desc='t_n'
    )

    CTDD_v_p_f = libpybiofeature.featurebuilder.build_CTDD_feature(
        path_to_fasta=path_dict['fasta']['v']['p'],
        seq_id_list=seq_id_dict['v']['p'],
        desc='v_p'
    )

    CTDD_v_n_f = libpybiofeature.featurebuilder.build_CTDD_feature(
        path_to_fasta=path_dict['fasta']['v']['n'],
        seq_id_list=seq_id_dict['v']['n'],
        desc='v_n'
    )

    onehot_t_p_f = libpybiofeature.featurebuilder.build_oneHot_feature(
        path_to_fasta=path_dict['fasta']['t']['p'],
        seq_id_list=seq_id_dict['t']['p'],
        desc='t_p',
        length=1000,
        cter=path_dict['model']['cter']
    )

    onehot_t_n_f = libpybiofeature.featurebuilder.build_oneHot_feature(
        path_to_fasta=path_dict['fasta']['t']['n'],
        seq_id_list=seq_id_dict['t']['n'],
        desc='t_n',
        length=1000,
        cter=path_dict['model']['cter']
    )

    onehot_v_p_f = libpybiofeature.featurebuilder.build_oneHot_feature(
        path_to_fasta=path_dict['fasta']['v']['p'],
        seq_id_list=seq_id_dict['v']['p'],
        desc='v_p',
        length=1000,
        cter=path_dict['model']['cter']
    )

    onehot_v_n_f = libpybiofeature.featurebuilder.build_oneHot_feature(
        path_to_fasta=path_dict['fasta']['v']['n'],
        seq_id_list=seq_id_dict['v']['n'],
        desc='v_n',
        length=1000,
        cter=path_dict['model']['cter']
    )

    aacpro_t_p_f = libpybiofeature.scratch_reader.accpro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['acctag']['t']['p'],
        seqid_list=seq_id_dict['t']['p'],
        length=1000,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    aacpro_t_n_f = libpybiofeature.scratch_reader.accpro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['acctag']['t']['n'],
        seqid_list=seq_id_dict['t']['n'],
        length=1000,
        dig_code=True,
        cter=path_dict['model']['cter']
    )
    aacpro_v_p_f = libpybiofeature.scratch_reader.accpro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['acctag']['v']['p'],
        seqid_list=seq_id_dict['v']['p'],
        length=1000,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    aacpro_v_n_f = libpybiofeature.scratch_reader.accpro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['acctag']['v']['n'],
        seqid_list=seq_id_dict['v']['n'],
        length=1000,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    ssapro_t_p_f = libpybiofeature.scratch_reader.sspro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['ssatag']['t']['p'],
        seqid_list=seq_id_dict['t']['p'],
        length=1000,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    ssapro_t_n_f = libpybiofeature.scratch_reader.sspro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['ssatag']['t']['n'],
        seqid_list=seq_id_dict['t']['n'],
        length=1000,
        dig_code=True,
        cter=path_dict['model']['cter']
    )
    ssapro_v_p_f = libpybiofeature.scratch_reader.sspro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['ssatag']['v']['p'],
        seqid_list=seq_id_dict['v']['p'],
        length=1000,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    ssapro_v_n_f = libpybiofeature.scratch_reader.sspro.get_func(
        path_to_json_db=path_dict['scratch']['db'],
        tag_name=path_dict['scratch']['ssatag']['v']['n'],
        seqid_list=seq_id_dict['v']['n'],
        length=1000,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    disopred3_t_p_f = libpybiofeature.disopred3.get_data(
        path_to_json=path_dict['DISOPRED3']['db']['index'],
        path_to_datadir=path_dict['DISOPRED3']['db']['file'],
        seqid_list=seq_id_dict['t']['p'],
        desc=path_dict['DISOPRED3']['desc']['t']['p'],
        length=1000,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    disopred3_t_n_f = libpybiofeature.disopred3.get_data(
        path_to_json=path_dict['DISOPRED3']['db']['index'],
        path_to_datadir=path_dict['DISOPRED3']['db']['file'],
        seqid_list=seq_id_dict['t']['n'],
        desc=path_dict['DISOPRED3']['desc']['t']['n'],
        length=1000,
        dig_code=True,
        cter=path_dict['model']['cter']
    )
    disopred3_v_p_f = libpybiofeature.disopred3.get_data(
        path_to_json=path_dict['DISOPRED3']['db']['index'],
        path_to_datadir=path_dict['DISOPRED3']['db']['file'],
        seqid_list=seq_id_dict['v']['p'],
        desc=path_dict['DISOPRED3']['desc']['v']['p'],
        length=1000,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    disopred3_v_n_f = libpybiofeature.disopred3.get_data(
        path_to_json=path_dict['DISOPRED3']['db']['index'],
        path_to_datadir=path_dict['DISOPRED3']['db']['file'],
        seqid_list=seq_id_dict['v']['n'],
        desc=path_dict['DISOPRED3']['desc']['v']['n'],
        length=1000,
        dig_code=True,
        cter=path_dict['model']['cter']
    )

    psssa_t_p_f = utils.ds_preprocess.merge_pd_list([
        aacpro_t_p_f, ssapro_t_p_f
    ])
    psssa_t_n_f = utils.ds_preprocess.merge_pd_list([
        aacpro_t_n_f, ssapro_t_n_f
    ])
    psssa_v_p_f = utils.ds_preprocess.merge_pd_list([
        aacpro_v_p_f, ssapro_v_p_f
    ])
    psssa_v_n_f = utils.ds_preprocess.merge_pd_list([
        aacpro_v_n_f, ssapro_v_n_f
    ])

    CTD_t_p_f = utils.ds_preprocess.merge_pd_list([
        CTDC_t_p_f, CTDT_t_p_f, CTDD_t_p_f
    ])
    CTD_t_n_f = utils.ds_preprocess.merge_pd_list([
        CTDC_t_n_f, CTDT_t_n_f, CTDD_t_n_f
    ])
    CTD_v_p_f = utils.ds_preprocess.merge_pd_list([
        CTDC_v_p_f, CTDT_v_p_f, CTDD_v_p_f
    ])
    CTD_v_n_f = utils.ds_preprocess.merge_pd_list([
        CTDC_v_n_f, CTDT_v_n_f, CTDD_v_n_f
    ])

    feature_dividend_list = utils.ds_preprocess.make_feature_dividend_list(
        f_length_list=[f.shape[1] for f in [
            psssa_t_p_f, aacpro_t_p_f, ssapro_t_p_f, onehot_t_p_f, pssm_t_p_f, possum_t_p_f, disopred3_t_p_f, aac_t_p_f, CTD_t_p_f
        ]]
    )
    model_selectsvm_list = [
        False, False, False, False, False, False, False, True, True,
    ]

    t_p_f = utils.ds_preprocess.merge_pd_list([
        psssa_t_p_f, aacpro_t_p_f, ssapro_t_p_f, onehot_t_p_f, pssm_t_p_f, possum_t_p_f, disopred3_t_p_f, aac_t_p_f, CTD_t_p_f
    ])
    t_n_f = utils.ds_preprocess.merge_pd_list([
        psssa_t_n_f, aacpro_t_n_f, ssapro_t_n_f, onehot_t_n_f, pssm_t_n_f, possum_t_n_f, disopred3_t_n_f, aac_t_n_f, CTD_t_n_f
    ])
    v_p_f = utils.ds_preprocess.merge_pd_list([
        psssa_v_p_f, aacpro_v_p_f, ssapro_v_p_f, onehot_v_p_f, pssm_v_p_f, possum_v_p_f, disopred3_v_p_f, aac_v_p_f, CTD_v_p_f
    ])
    v_n_f = utils.ds_preprocess.merge_pd_list([
        psssa_v_n_f, aacpro_v_n_f, ssapro_v_n_f, onehot_v_n_f, pssm_v_n_f, possum_v_n_f, disopred3_v_n_f, aac_v_n_f, CTD_v_n_f
    ])

    t_p_f.columns = list(range(len(t_p_f.columns)))
    t_n_f.columns = list(range(len(t_n_f.columns)))
    v_p_f.columns = list(range(len(v_p_f.columns)))
    v_n_f.columns = list(range(len(v_n_f.columns)))

    t_p_l = utils.ds_preprocess.make_binary_label(
        size=t_p_f.shape[0], label=True)
    t_n_l = utils.ds_preprocess.make_binary_label(
        size=t_n_f.shape[0], label=False)

    v_p_l = utils.ds_preprocess.make_binary_label(
        size=v_p_f.shape[0], label=True)
    v_n_l = utils.ds_preprocess.make_binary_label(
        size=v_n_f.shape[0], label=False)

    t_5C = utils.ds_preprocess.get_5C_data(
        shape_to_chiose=t_p_f.shape[0],
        t_p_f=t_p_f,
        t_p_l=t_p_l,
        t_n_f=t_n_f,
        t_n_l=t_n_l,
        shufflesplit={
            'shufflesplit_index_file': path_dict['shufflesplit_index_file']
        } if path_dict['model']['size'] == 'small' else None
    )

    t_f, t_l = utils.ds_preprocess.make_merge(
        t_p_f=t_p_f,
        t_p_l=t_p_l,
        t_n_f=t_n_f,
        t_n_l=t_n_l,
    )

    v_f, v_l = utils.ds_preprocess.make_merge(
        t_p_f=v_p_f,
        t_p_l=v_p_l,
        t_n_f=v_n_f,
        t_n_l=v_n_l,
    )

    model_cv = 10
    if path_dict['model']['size'] == 'small':
        model_cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=0)

    if only_tt_model == False:
        Five_Cross_Get_model(
            feature_dividend_list=feature_dividend_list,
            model_selectsvm_list=model_selectsvm_list,
            model_construct_funtion=functools.partial(
                CNNT4SE_Model, desc=path_dict['model']['cv']['desc'], cv=model_cv),
            t_5C=t_5C,
            v_f=v_f,
            v_l=v_l,
            path_to_model_pickle=path_dict['model']['cv']['model_pickle'],
            path_to_model_result=path_dict['model']['cv']['model_result'],
            size_of_data=path_dict['model']['size']
        )
    Five_Cross_Get_model(
        feature_dividend_list=feature_dividend_list,
        model_selectsvm_list=model_selectsvm_list,
        model_construct_funtion=functools.partial(
            CNNT4SE_Model, desc=path_dict['model']['tt']['desc'], cv=model_cv),
        t_5C=(((t_f, t_l),
               (v_f, v_l)),),
        v_f=v_f,
        v_l=v_l,
        path_to_model_pickle=path_dict['model']['tt']['model_pickle'],
        path_to_model_result=path_dict['model']['tt']['model_result'],
        size_of_data=None
    )
