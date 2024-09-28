'''
Author: George Zhao
Date: 2022-03-12 12:08:04
LastEditors: George Zhao
LastEditTime: 2022-03-12 12:28:23
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
import os
import re
import hashlib


def sha256sum(filename):
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


def sha256_go(path_to_root: str,):
    path_to_root = os.path.join(path_to_root, '')
    path_stack = [path_to_root, ]
    path_of_result_list = list()
    while len(path_stack) != 0:
        path_current = path_stack.pop()
        if os.path.isdir(path_current) == True:
            path_stack.extend([os.path.join(path_current, item)
                               for item in os.listdir(path_current)])
            continue
        else:
            for regx_pattern in [r'^(.+\.pkl)$', r'^(.+\.h5)$', r'^(.+\.model)$', r'^(.+\.arff)$', r'^(.+\.bin)$']:
                reresult = re.findall(
                    regx_pattern, path_current)
                if len(reresult) != 1:
                    continue
                else:
                    path_of_result_list.append(reresult[0])
                    break

    motifed_file = list()
    for resultjson_path in path_of_result_list:
        sumfile_name = resultjson_path + '.sha256sum'
        old_sum = None
        if os.path.exists(sumfile_name) == True:
            with open(sumfile_name, 'r', encoding='UTF-8')as f:
                old_sum = f.read()
        new_sum = sha256sum(resultjson_path)
        with open(sumfile_name, 'w+', encoding='UTF-8')as f:
            f.write(new_sum)
        if old_sum != new_sum:
            motifed_file.append(
                {'path': resultjson_path, 'old': old_sum, 'new': new_sum})
    return motifed_file
