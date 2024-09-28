'''
Author: George Zhao
Date: 2022-03-04 11:33:42
LastEditors: George Zhao
LastEditTime: 2022-06-23 21:56:42
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
# %%
import os
import sys
import pickle
import json
import functools
sys.path.append('src')

import utils
import libpybiofeature

from libmodel import common

import tensorflow as tf

epochs = 100
batch_size = 80
learning_rate = 0.005
momentum = 0.9
early_stop_iter = 10


def get_model():
    input1 = tf.keras.layers.Input(shape=(2000, ), name='Input_Layer')
    reshape_layer = tf.keras.layers.Reshape(
        (100, 20, 1), name='reshape_layer')(input1)

    conv1 = tf.keras.layers.Conv2D(
        50, (20, 12), activation=tf.keras.activations.relu)(reshape_layer)

    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(conv1)

    flatten_layer = tf.keras.layers.Flatten()(pool1)

    batch_norm_layer = tf.keras.layers.BatchNormalization()(flatten_layer)

    dense1 = tf.keras.layers.Dense(
        650, activation=tf.keras.activations.relu)(batch_norm_layer)

    dropout1 = tf.keras.layers.Dropout(0.5)(dense1)

    output_layer = tf.keras.layers.Dense(
        1, activation=tf.keras.activations.sigmoid)(dropout1)

    model = tf.keras.models.Model(
        inputs=input1, outputs=output_layer, name='DeepT3')

    model.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum
        ),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=[
            tf.keras.metrics.binary_accuracy,
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    return model


class DeepT3_Model(common.Model_Final):
    def __init__(self, cv, desc):
        super().__init__(cv, desc=desc)
        self.model = None

    def tranmodel(self, f, l):
        super().tranmodel(f, l)
        self.model = get_model()
        self.model.fit(
            f,
            tf.constant(l),
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='auc', min_delta=0, patience=10, verbose=0,
                    mode='auto', baseline=None, restore_best_weights=True
                ),
            ]
        )
        return self

    def predict(self, f):
        super().predict(f)
        return self.model.predict(f)

    def save_to_file(self, path_to_dir, firstname=None):
        super().save_to_file(path_to_dir)
        self.model.save(
            self.filename_code(path_to_dir, firstname)
        )
        return self

    def load_model(self, path_to_dir, firstname=None):
        super().load_model(path_to_dir)
        self.model = tf.keras.models.load_model(
            self.filename_code(path_to_dir, firstname)
        )
        return self

    def clean_model(self):
        super().clean_model()
        self.model = None
        return self

    def filename_code(self, path_to_dir, firstname):
        path_to_dir = os.path.splitext(path_to_dir)[0]
        if firstname is not None:
            if os.path.exists(os.path.join(path_to_dir, self.desc)) == False:
                os.path.join(path_to_dir, self.desc)
            return os.path.join(path_to_dir, *[self.desc, f'{firstname}.h5'])
        return os.path.join(path_to_dir, self.desc)


def Five_Cross_Get_model(
    model_construct_funtion,
    Five_Cross_set: list,
    v_f,
    v_l,
    path_to_model_pickle: str,
    path_to_model_result: str,
    size_of_data: str,
):
    model_set = list()
    model_result_set = list()
    for i in range(len(Five_Cross_set)):

        train, test = Five_Cross_set[i]
        validated_f = v_f

        model: DeepT3_Model = model_construct_funtion()
        model.tranmodel(train[0], train[1])

        model_set.append(model)

        model_result_set.append(
            {
                "training": {
                    "origin": {
                        f'pred': list(model.predict(train[0])),
                        f'label': list(train[1])},
                    "evaluation": {
                    },
                    "option": {
                    }
                },
                "testing": {
                    "origin": {
                        f'pred': list(model.predict(test[0])),
                        f'label': list(test[1])},
                    "evaluation": {
                    },
                    "option": {
                    }
                },
                "validated": {
                    "origin": {
                        f'pred': list(model.predict(validated_f)),
                        f'label': list(v_l)},
                    "evaluation": {
                    },
                    "option": {
                    }
                },
                "detail": {
                    "model": 'DeepT3',
                    'desc': model.desc,
                    'iteration': i,
                    "size_of_data": size_of_data,
                }
            }
        )
        model.save_to_file(
            path_to_dir=path_to_model_pickle,
            firstname=f'{i}'
        ).clean_model()

    model_result_set = utils.ds_preprocess.Five_Cross_Evaluation(
        model_result_set,
        pro_cutoff=0.5,
        mode='loo' if size_of_data == 'small' else None
    )

    if os.path.exists(os.path.split(path_to_model_pickle)[0]) == False:
        os.makedirs(os.path.split(path_to_model_pickle)[0])
    if os.path.exists(os.path.split(path_to_model_result)[0]) == False:
        os.makedirs(os.path.split(path_to_model_result)[0])

    with open(path_to_model_pickle, 'bw+') as f:
        pickle.dump(model_set, f)

    with open(path_to_model_result, 'w+', encoding='UTF-8') as f:
        json.dump(model_result_set, f, cls=utils.ds_preprocess.MyEncoder)

    return model_set, model_result_set


def research(path_dict: dict):
    seq_id_dict = None
    with open(path_dict['seq_id'], 'r', encoding='UTF-8') as f:
        seq_id_dict = json.load(f)

    t_p_f = libpybiofeature.featurebuilder.build_oneHot_feature(
        path_to_fasta=path_dict['onehot']['t']['p'],
        seq_id_list=seq_id_dict['t']['p'],
        desc='t_p',
        length=100,
        cter=path_dict['onehot']['cter']
    )

    t_n_f = libpybiofeature.featurebuilder.build_oneHot_feature(
        path_to_fasta=path_dict['onehot']['t']['n'],
        seq_id_list=seq_id_dict['t']['n'],
        desc='t_n',
        length=100,
        cter=path_dict['onehot']['cter']
    )

    v_p_f = libpybiofeature.featurebuilder.build_oneHot_feature(
        path_to_fasta=path_dict['onehot']['v']['p'],
        seq_id_list=seq_id_dict['v']['p'],
        desc='v_p',
        length=100,
        cter=path_dict['onehot']['cter']
    )

    v_n_f = libpybiofeature.featurebuilder.build_oneHot_feature(
        path_to_fasta=path_dict['onehot']['v']['n'],
        seq_id_list=seq_id_dict['v']['n'],
        desc='v_n',
        length=100,
        cter=path_dict['onehot']['cter']
    )

    t_p_l = utils.ds_preprocess.make_binary_label(
        size=t_p_f.shape[0], label=True)
    t_n_l = utils.ds_preprocess.make_binary_label(
        size=t_n_f.shape[0], label=False)

    v_p_l = utils.ds_preprocess.make_binary_label(
        size=v_p_f.shape[0], label=True)
    v_n_l = utils.ds_preprocess.make_binary_label(
        size=v_n_f.shape[0], label=False)

    t_f, t_l = utils.ds_preprocess.make_merge(t_p_f, t_p_l, t_n_f, t_n_l)
    v_f, v_l = utils.ds_preprocess.make_merge(v_p_f, v_p_l, v_n_f, v_n_l)

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

    Five_Cross_Get_model(
        model_construct_funtion=functools.partial(
            DeepT3_Model, desc=path_dict['model']['cv']['desc'], cv=None),
        Five_Cross_set=t_5C,
        v_f=v_f,
        v_l=v_l,
        path_to_model_pickle=path_dict['model']['cv']['model_pickle'],
        path_to_model_result=path_dict['model']['cv']['model_result'],
        size_of_data=path_dict['model']['size']
    )

    Five_Cross_Get_model(
        model_construct_funtion=functools.partial(
            DeepT3_Model, desc=path_dict['model']['tt']['desc'], cv=None),
        Five_Cross_set=(([t_f, t_l], [v_f, v_l]),),
        v_f=v_f,
        v_l=v_l,
        path_to_model_pickle=path_dict['model']['tt']['model_pickle'],
        path_to_model_result=path_dict['model']['tt']['model_result'],
        size_of_data=None
    )
