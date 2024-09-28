'''
Author: George Zhao
Date: 2021-08-02 14:56:21
LastEditors: George Zhao
LastEditTime: 2021-08-02 15:26:10
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
# %%
default_aaorder = {
    'alphabetically': 'ACDEFGHIKLMNPQRSTVWY',
    'polarity': 'DENKRQHSGTAPYVMCWIFL',
    'sideChainVolume': 'GASDPCTNEVHQILMKRFYW',
}


def get_cal_param(gap=5, aaorder: str = default_aaorder['alphabetically']):
    param = {'aaorder': aaorder}

    aaPairs = []
    for aa1 in aaorder:
        for aa2 in aaorder:
            aaPairs.append(aa1 + aa2)
    param.update({'aaPairs': aaPairs})

    header = []
    for g in range(gap + 1):
        for aa in aaPairs:
            header.append(aa + '.gap' + str(g))
    param.update({'header': header})
    return param


defalut_param = get_cal_param()


def CKSAAP(seq_aa: str, desc='undefine', gap=5, buildin_param: dict = defalut_param):
    if gap < 0:
        raise ValueError(
            f'CKSAAP: gap should be greater than or equal to 0 (Get {gap}).')

    if len(seq_aa) < gap + 2:
        raise ValueError(
            f'CKSAAP: Input sequences:{desc} should be greater than or equal to gap({gap}) + 2 (Get {len(seq_aa)}).')

    code = list()
    for g in range(gap + 1):
        myDict = {}
        for pair in buildin_param['aaPairs']:
            myDict[pair] = 0
        sum = 0
        for index1 in range(len(seq_aa)):
            index2 = index1 + g + 1
            if index1 < len(seq_aa) and index2 < len(seq_aa) and seq_aa[index1] in buildin_param['aaorder'] and seq_aa[index2] in buildin_param['aaorder']:
                myDict[seq_aa[index1] + seq_aa[index2]
                       ] = myDict[seq_aa[index1] + seq_aa[index2]] + 1
                sum = sum + 1
        for pair in buildin_param['aaPairs']:
            code.append(myDict[pair] / sum)
    return code
