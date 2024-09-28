import os
import sys
sys.path.append("src")

import warnings
warnings.filterwarnings('ignore')

import json

from libfeatureselection import feature_loader
from libfeatureselection import selection
from libfeatureselection import tsne

import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

if __name__ == "__main__":
    import utils
    work_Dir = utils.workdir.workdir(os.getcwd(), 4)

    prot_type_infor = [
        (1, True, 'small'),
        (2, False, 'small'),
        (3, False, 'big'),
        (4, True, 'big'),
        (6, False, 'small'),
    ]

    if "TYPE_TO_ANALYSIS" in os.environ and os.environ['TYPE_TO_ANALYSIS'] != "":
        prot_type_infor = [
            item
            for item in prot_type_infor
            if item[0] == int(os.environ['TYPE_TO_ANALYSIS'])
        ]

    print(
        f"Feature Selection for: {', '.join([str(item[0]) for item in prot_type_infor])}"
    )

    for protype, cter_bool, db_size in prot_type_infor:
        Tx_arg = {
            "type": f'T{protype}',
            'seq_id': os.path.join(work_Dir, *['data', 'db', f'T{protype}', 'seq_id.json']),
            'shufflesplit_index_file': os.path.join(work_Dir, *['data', 'db', f'T{protype}', 'seq_id_shufflesplit.json']),
            'fasta': {
                'cter': cter_bool,
                't': {
                    'p': os.path.join(work_Dir, *['data', 'db', f'T{protype}', 't_p.fasta']),
                    'n': os.path.join(work_Dir, *['data', 'db', f'T{protype}', 't_n.fasta'])
                },
                'v': {
                    'p': os.path.join(work_Dir, *['data', 'db', f'T{protype}', 'v_p.fasta']),
                    'n': os.path.join(work_Dir, *['data', 'db', f'T{protype}', 'v_n.fasta'])
                },
            },
            'expasy': {
                "t": {
                    'p': os.path.join(work_Dir, *['out', f'T{protype}', 'data', f'expasy.t{protype}.t_p.json']),
                    "n": os.path.join(work_Dir, *['out', f'T{protype}', 'data', f'expasy.t{protype}.t_n.json'])
                },
                "v": {
                    'p': os.path.join(work_Dir, *['out', f'T{protype}', 'data', f'expasy.t{protype}.v_p.json']),
                    "n": os.path.join(work_Dir, *['out', f'T{protype}', 'data', f'expasy.t{protype}.v_n.json'])
                }
            },
            'bliulab': {
                't': {
                    'idna': os.path.join(work_Dir, *['out', f'T{protype}', 'data', 'iDNA_Prot_dis_data_t.json', ]),
                    'top': os.path.join(work_Dir, *['out', f'T{protype}', 'data', 'Top_n_gram_data_t.json', ]),
                    'pse': os.path.join(work_Dir, *['out', f'T{protype}', 'data', 'PC_PSEACC_data_t.json', ]),
                    'SC': os.path.join(work_Dir, *['out', f'T{protype}', 'data', 'SC_PSEACC_data_t.json', ]),
                    'PC': os.path.join(work_Dir, *['out', f'T{protype}', 'data', 'PC_PSEACC_data_t.json', ]),
                },
                'v': {
                    'idna': os.path.join(work_Dir, *['out', f'T{protype}', 'data', 'iDNA_Prot_dis_data_v.json', ]),
                    'top': os.path.join(work_Dir, *['out', f'T{protype}', 'data', 'Top_n_gram_data_v.json', ]),
                    'pse': os.path.join(work_Dir, *['out', f'T{protype}', 'data', 'PC_PSEACC_data_v.json', ]),
                    'SC': os.path.join(work_Dir, *['out', f'T{protype}', 'data', 'SC_PSEACC_data_v.json', ]),
                    'PC': os.path.join(work_Dir, *['out', f'T{protype}', 'data', 'PC_PSEACC_data_v.json', ]),
                },
            },
            "ss": "{work_Dir}/out/T{protype}/data/bert/{{db_type}}_ss_{cter}.pkl".format(
                work_Dir=work_Dir,
                protype=protype,
                cter="c" if cter_bool else "n"
            ),
            "sa": "{work_Dir}/out/T{protype}/data/bert/{{db_type}}_sa_{cter}.pkl".format(
                work_Dir=work_Dir,
                protype=protype,
                cter="c" if cter_bool else "n"
            ),
            "diso": "{work_Dir}/out/T{protype}/data/bert/{{db_type}}_diso_{cter}.pkl".format(
                work_Dir=work_Dir,
                protype=protype,
                cter="c" if cter_bool else "n"
            ),
            "rawuntrain": "{work_Dir}/out/T{protype}/data/bert/{{db_type}}_rawuntrain_{cter}.pkl".format(
                work_Dir=work_Dir,
                protype=protype,
                cter="c" if cter_bool else "n"
            ),
            'a': {
                't': {
                    "p": os.path.join(work_Dir, *['out', f'T{protype}', 'data', f'aligned_t{protype}_t_p.json', ]),
                    "n": os.path.join(work_Dir, *['out', f'T{protype}', 'data', f'aligned_t{protype}_t_n.json', ]),
                },
                'v': {
                    "p": os.path.join(work_Dir, *['out', f'T{protype}', 'data', f'aligned_t{protype}_v_p.json', ]),
                    "n": os.path.join(work_Dir, *['out', f'T{protype}', 'data', f'aligned_t{protype}_v_n.json', ]),
                }
            },
            'possum': {
                'index': os.path.join(work_Dir, *['out', f'T{protype}', 'data', 'possum', 'possum_index.json']),
                'fasta_pattern': os.path.join(work_Dir, *['data', 'db', f'T{protype}', '{taskname}.fasta']),
                'pssm_db_pattern': os.path.join(work_Dir, *['out', f'T{protype}', 'data', 'possum', '{zipid}_pssm_features.zip']),
                'pssm_fdb_pattern': os.path.join(work_Dir, *['out', f'T{protype}', 'data', 'possum', '{zipid}_pssm_features.zip']),
                'pssm_rdb_pattern': os.path.join(work_Dir, *['out', f'T{protype}', 'data', 'possum', '{zipid}_pssm_files.zip'])
            },
            'model': {
                'size': db_size,
                'cter': cter_bool,
                "path_to_save_dir": f"out/libfeatureselection/T{protype}/model/"
            }
        }

        dataset = feature_loader.load_feature(
            TxSE_args=Tx_arg
        )

        n_jobs = (
            (os.cpu_count() - 2)
            if "n_jobs" not in os.environ or os.environ['n_jobs'] == "" else
            int(os.environ['n_jobs'])
        )

        print(f"n_jobs: {n_jobs}")

        tsne.feature_2d_plot_for_dataset(
            dataset=dataset,
            path_to_out_dir=f"out/libfeatureselection/T{protype}/feature_plot",
            n_jobs=n_jobs
        )

        with open(f"{Tx_arg['model']['path_to_save_dir']}/feature_dim.json", "w+", encoding='UTF-8') as f:
            json.dump({
                item['name']: item['t_p'].shape[1]
                for item in dataset
            }, f)

        # feature_prob = (
        #     0.5
        #     if "FEATURE_PROB" not in os.environ or os.environ['FEATURE_PROB'] == "" else
        #     float(os.environ['FEATURE_PROB'])
        # )

        # assert feature_prob < 1 and feature_prob > 0

        # print(f"feature_prob: {feature_prob}")

        # selector = selection.FeatureGroupSelection(
        #     desc=f"T{protype}"
        # ).search(
        #     ordered_data=dataset,
        #     path_to_save_dir=Tx_arg['model']['path_to_save_dir'],
        #     n_jobs=n_jobs,
        #     feature_prob=feature_prob
        # )

        # with open(f"{Tx_arg['model']['path_to_save_dir']}/feature_order.json", "w+", encoding='UTF-8') as f:
        #     json.dump(selector.feature_order, f)
