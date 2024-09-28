'''
Author: George Zhao
Date: 2021-05-29 15:13:51
LastEditors: George Zhao
LastEditTime: 2021-05-29 15:44:00
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
# %%
import os
import sys
sys.path.append('../..')
# %%
import utils

work_Dir = utils.workdir.workdir(os.getcwd(), 3)
rdata_Dir = os.path.join(work_Dir, 'data')

logger = utils.glogger.Glogger('EP3_Get_substitution_matrices', os.path.join(
    work_Dir, 'log'))
log_wrapper = utils.glogger.log_wrapper(logger=logger)

lib_Dir = os.path.join(work_Dir, 'lib')
# %%
from Bio.Align import substitution_matrices
# %%
@log_wrapper
def Build_substitution_matrices(name: str):
    mat = None
    with open(os.path.join(lib_Dir, f'substitution_matrices/{name}')) as f:
        mat = substitution_matrices.read(f, dtype=float)
    return mat
