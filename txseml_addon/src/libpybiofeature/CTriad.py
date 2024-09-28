'''
Author: George Zhao
Date: 2021-08-02 13:35:09
LastEditors: George Zhao
LastEditTime: 2021-08-02 14:51:23
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
AAGroup = {
    'g1': 'AGV',
    'g2': 'ILFP',
    'g3': 'YMTS',
    'g4': 'HNQW',
    'g5': 'RK',
    'g6': 'DE',
    'g7': 'C'
}

_myGroups = sorted(AAGroup.keys())

AADict = {}
for g in _myGroups:
    for aa in AAGroup[g]:
        AADict[aa] = g

features = [f1 + '.' + f2 + '.' +
            f3 for f1 in _myGroups for f2 in _myGroups for f3 in _myGroups]


def _CalculateKSCTriad(sequence, gap=0, features=features, AADict=AADict):
    res = []
    for g in range(gap + 1):
        myDict = {}
        for f in features:
            myDict[f] = 0

        for i in range(len(sequence)):
            if i + gap + 1 < len(sequence) and i + 2 * gap + 2 < len(sequence):
                fea = AADict[sequence[i]] + '.' + AADict[sequence[i +
                                                                  gap + 1]] + '.' + AADict[sequence[i + 2 * gap + 2]]
                myDict[fea] = myDict[fea] + 1

        maxValue, minValue = max(myDict.values()), min(myDict.values())
        for f in features:
            res.append((myDict[f] - minValue) / maxValue)

    return res


def CTriad(seq_aa: str, desc='undefine', gap=0):

    if len(seq_aa) < 3:
        raise ValueError(
            f'CTriad: Input sequences:{desc} should be greater than 3 (Get {len(seq_aa)}).')

    return _CalculateKSCTriad(seq_aa, gap, features, AADict)
