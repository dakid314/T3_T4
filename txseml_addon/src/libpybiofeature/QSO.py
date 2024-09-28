'''
Author: George Zhao
Date: 2021-08-03 16:15:30
LastEditors: George Zhao
LastEditTime: 2022-02-27 23:20:49
Description:
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import sys
sys.path.append('/mnt/md0/Public/一站式/txseml_backend_addon/txseml_addon/src/')
import os
from utils import workdir
import sys
lib_Dir = os.path.join(
    workdir.workdir(os.getcwd(), 3), 'lib'
)

import numpy as np

dataFile = os.path.join(
    lib_Dir, *['libpybiofeature', 'Schneider-Wrede.txt']
)
dataFile1 = os.path.join(
    lib_Dir, *['libpybiofeature', 'Grantham.txt']
)
AA = 'ACDEFGHIKLMNPQRSTVWY'
AA1 = 'ARNDCQEGHILKMFPSTWYV'
DictAA = {}
for i in range(len(AA)):
    DictAA[AA[i]] = i
DictAA1 = {}
for i in range(len(AA1)):
    DictAA1[AA1[i]] = i

with open(dataFile) as f:
    records = f.readlines()[1:]
AADistance = []
for i in records:
    array = i.rstrip().split()[1:] if i.rstrip() != '' else None
    AADistance.append(array)
AADistance = np.array(
    [float(AADistance[i][j]) for i in range(len(AADistance)) for j in range(len(AADistance[i]))]).reshape((20, 20))

with open(dataFile1) as f:
    records = f.readlines()[1:]
AADistance1 = []
for i in records:
    array = i.rstrip().split()[1:] if i.rstrip() != '' else None
    AADistance1.append(array)
AADistance1 = np.array(
    [float(AADistance1[i][j]) for i in range(len(AADistance1)) for j in range(len(AADistance1[i]))]).reshape(
    (20, 20))


def get_header(nlag=30):
    header = []
    for aa in AA1:
        header.append('Schneider.Xr.' + aa)
    for aa in AA1:
        header.append('Grantham.Xr.' + aa)
    for n in range(1, nlag + 1):
        header.append('Schneider.Xd.' + str(n))
    for n in range(1, nlag + 1):
        header.append('Grantham.Xd.' + str(n))
    return header


default_header = get_header()


def QSOrder(seq_aa: str, desc='undefine', nlag=30, w=0.1, **kw):
    if len(seq_aa) < nlag + 1:
        raise ValueError(
            f'QSOrder: Sequence length should be larger than the nlag+1({nlag + 1}): Get({len(seq_aa)})')

    code = list()
    arraySW = []
    arrayGM = []
    for n in range(1, nlag + 1):
        arraySW.append(
            sum([AADistance[DictAA[seq_aa[j]]][DictAA[seq_aa[j + n]]] ** 2 for j in range(len(seq_aa) - n)]))
        arrayGM.append(sum(
            [AADistance1[DictAA1[seq_aa[j]]][DictAA1[seq_aa[j + n]]] ** 2 for j in range(len(seq_aa) - n)]))
    myDict = {}
    for aa in AA1:
        myDict[aa] = seq_aa.count(aa)
    for aa in AA1:
        code.append(myDict[aa] / (1 + w * sum(arraySW)))
    for aa in AA1:
        code.append(myDict[aa] / (1 + w * sum(arrayGM)))
    for num in arraySW:
        code.append((w * num) / (1 + w * sum(arraySW)))
    for num in arrayGM:
        code.append((w * num) / (1 + w * sum(arrayGM)))
    return code

