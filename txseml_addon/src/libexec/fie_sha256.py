'''
Author: George Zhao
Date: 2021-08-04 18:43:02
LastEditors: George Zhao
LastEditTime: 2022-03-12 14:42:45
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import sys
sys.path.append('src')
import os
import json
import utils

work_Dir = utils.workdir.workdir(os.getcwd(), 3)

with open('filesum.report.json', 'w+', encoding='UTF-8') as f:
    json.dump({'T1': utils.model_sha256sum.sha256_go(
        path_to_root=os.path.join(work_Dir, *[
            'out', 'T1', 'model'
        ])
    ),

        'T2': utils.model_sha256sum.sha256_go(
        path_to_root=os.path.join(work_Dir, *[
            'out', 'T2', 'model'
        ])
    ),

        'T3': utils.model_sha256sum.sha256_go(
        path_to_root=os.path.join(work_Dir, *[
            'out', 'T3', 'model'
        ])
    ),

        'T4': utils.model_sha256sum.sha256_go(
        path_to_root=os.path.join(work_Dir, *[
            'out', 'T4', 'model'
        ])
    ),

        'T6': utils.model_sha256sum.sha256_go(
        path_to_root=os.path.join(work_Dir, *[
            'out', 'T6', 'model'
        ])
    ),
    }, f)
