'''
Author: George Zhao
Date: 2021-08-04 18:43:02
LastEditors: George Zhao
LastEditTime: 2022-06-29 11:23:38
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import sys
sys.path.append('src')
import os
import utils

work_Dir = utils.workdir.workdir(os.getcwd(), 3)
work_Dir = os.path.join(work_Dir, *["..", "TxSEpp_report"])

# T1
utils.model_reporter.get_md_report(
    path_to_root=os.path.join(work_Dir, *[
        'out', 'T1', 'model'
    ]),
    path_to_out=os.path.join(work_Dir, *[
        'tmp', 'data_out_md_docs', 'docs', 'model_report_T1.html'
    ]),
    optimal=True
)

# T2
utils.model_reporter.get_md_report(
    path_to_root=os.path.join(work_Dir, *[
        'out', 'T2', 'model'
    ]),
    path_to_out=os.path.join(work_Dir, *[
        'tmp', 'data_out_md_docs', 'docs', 'model_report_T2.html'
    ]),
    optimal=True
)

# T3
utils.model_reporter.get_md_report(
    path_to_root=os.path.join(work_Dir, *[
        'out', 'T3', 'model'
    ]),
    path_to_out=os.path.join(work_Dir, *[
        'tmp', 'data_out_md_docs', 'docs', 'model_report_T3.html'
    ]),
    optimal=True
)

# T4
utils.model_reporter.get_md_report(
    path_to_root=os.path.join(work_Dir, *[
        'out', 'T4', 'model'
    ]),
    path_to_out=os.path.join(work_Dir, *[
        'tmp', 'data_out_md_docs', 'docs', 'model_report_T4.html'
    ]),
    optimal=True
)

# T6
utils.model_reporter.get_md_report(
    path_to_root=os.path.join(work_Dir, *[
        'out', 'T6', 'model'
    ]),
    path_to_out=os.path.join(work_Dir, *[
        'tmp', 'data_out_md_docs', 'docs', 'model_report_T6.html'
    ]),
    optimal=True
)
