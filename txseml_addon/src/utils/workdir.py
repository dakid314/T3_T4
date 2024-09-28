'''
Author: George Zhao
Date: 2021-03-20 11:04:24
LastEditors: George Zhao
LastEditTime: 2022-01-20 22:20:48
Description:
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import os


def workdir(path: str, time: int = 2):
    originpath = path
    for _ in range(time + 1):
        if os.path.exists(os.path.join(path, '.targetdir')) == True:
            return os.path.abspath(path)
        else:
            path = os.path.join(path, '..')
    raise RuntimeError(f'Not Found targetdir: {originpath}')
