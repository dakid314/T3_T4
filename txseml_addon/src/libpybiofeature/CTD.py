'''
Author: George Zhao
Date: 2021-08-02 16:31:27
LastEditors: George Zhao
LastEditTime: 2021-08-07 15:07:17
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import math


class CTD_property:
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity': 'LIFWCMVY',
        'polarizability': 'GASDT',
        'charge': 'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess': 'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity': 'PATGS',
        'polarizability': 'CPNVEQIL',
        'charge': 'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess': 'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity': 'HQRKNED',
        'polarizability': 'KMHFRYW',
        'charge': 'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess': 'MSPTHY'
    }

    groups = [group1, group2, group3]
    property = (
        'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
        'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
        'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')


class _CTDT:

    def get_header():
        header = list()
        for p in CTD_property.property:
            for tr in ('Tr1221', 'Tr1331', 'Tr2332'):
                header.append(p + '.' + tr)
        return header

    header = get_header()


class _CTDC:

    def get_header():
        header = list()
        for p in CTD_property.property:
            for g in range(1, len(CTD_property.groups) + 1):
                header.append(p + '.G' + str(g))
        return header

    header = get_header()

    @staticmethod
    def Count(seq1, seq2):
        sum = 0
        for aa in seq1:
            sum = sum + seq2.count(aa)
        return sum


class _CTDD:

    def get_header():
        header = list()
        for p in CTD_property.property:
            for g in ('1', '2', '3'):
                for d in ['0', '25', '50', '75', '100']:
                    header.append(p + '.' + g + '.residue' + d)
        return header

    header = get_header()

    @staticmethod
    def Count(aaSet, sequence):
        number = 0
        for aa in sequence:
            if aa in aaSet:
                number = number + 1
        cutoffNums = [1, math.floor(
            0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
        cutoffNums = [i if i >= 1 else 1 for i in cutoffNums]

        code = []
        for cutoff in cutoffNums:
            myCount = 0
            for i in range(len(sequence)):
                if sequence[i] in aaSet:
                    myCount += 1
                    if myCount == cutoff:
                        code.append((i + 1) / len(sequence) * 100)
                        break
            if myCount == 0:
                code.append(0)
        return code


CTDT_localobj = _CTDT()
CTDC_localobj = _CTDC()
CTDD_localobj = _CTDD()


def CTDC(seq_aa: str, desc='undefine'):
    code = list()
    for p in CTD_property.property:
        c1 = CTDC_localobj.Count(
            CTD_property.group1[p], seq_aa) / len(seq_aa)
        c2 = CTDC_localobj.Count(
            CTD_property.group2[p], seq_aa) / len(seq_aa)
        c3 = 1 - c1 - c2
        code = code + [c1, c2, c3]
    return code


def CTDD(seq_aa: str, desc='undefine'):

    code = list()
    for p in CTD_property.property:
        code = code + _CTDD.Count(CTD_property.group1[p], seq_aa) + \
            _CTDD.Count(CTD_property.group2[p], seq_aa) + \
            _CTDD.Count(CTD_property.group3[p], seq_aa)
    return code


def CTDT(seq_aa: str, desc='undefine'):
    code = list()
    aaPair = [seq_aa[j:j + 2] for j in range(len(seq_aa) - 1)]
    for p in CTD_property.property:
        c1221, c1331, c2332 = 0, 0, 0
        for pair in aaPair:
            if (pair[0] in CTD_property.group1[p] and pair[1] in CTD_property.group2[p]) or (pair[0] in CTD_property.group2[p] and pair[1] in CTD_property.group1[p]):
                c1221 = c1221 + 1
                continue
            if (pair[0] in CTD_property.group1[p] and pair[1] in CTD_property.group3[p]) or (pair[0] in CTD_property.group3[p] and pair[1] in CTD_property.group1[p]):
                c1331 = c1331 + 1
                continue
            if (pair[0] in CTD_property.group2[p] and pair[1] in CTD_property.group3[p]) or (pair[0] in CTD_property.group3[p] and pair[1] in CTD_property.group2[p]):
                c2332 = c2332 + 1
        code = code + \
            [c1221 / len(aaPair), c1331 / len(aaPair), c2332 / len(aaPair)]
    return code
