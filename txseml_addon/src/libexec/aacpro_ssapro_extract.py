'''
Author: George Zhao
Date: 2022-05-21 11:05:58
LastEditors: George Zhao
LastEditTime: 2022-08-27 10:06:15
Description: T1SEstacker used.
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
# %%
import json
import itertools

# %%


def _extract_from_file(path_to_json: str, tagname: str):
    jsondb = None
    with open(path_to_json, "r", encoding='UTF-8') as f:
        jsondb = json.load(f)
    k_list = [k for k in jsondb['data'].keys() if k.find(tagname) == 0]
    db_choise_by_k = list(itertools.chain(
        *[jsondb['data'][k] for k in k_list]))
    return db_choise_by_k


def extract_from_files(path_to_json: list, path_to_out: str, tagname_list: str, cterreverse: bool):
    with open(path_to_out, "w+", encoding="UTF-8") as f:
        f.write("".join([
            f">{seqitem['id']}\n{seqitem['data'].splitlines()[2].upper().replace('B', 'A') if cterreverse == False else ''.join(reversed(seqitem['data'].splitlines()[2].upper().replace('B', 'A'))) }\n"
            for seqitem in
            list(itertools.chain(*[
                _extract_from_file(path_to_json=path_to_json, tagname=tagname)
                for tagname in tagname_list
            ]))]))
    return


if __name__ == "__main__":
    import os
    import sys
    sys.path.append('..')
    sys.path.append('src')
    from utils import workdir
    work_Dir = workdir.workdir(os.getcwd(), 3)

    for protype, ter in [(1, "c"), (2, "n"), (3, "n"), (4, "c"), (6, "n")]:
        config = {
            'scratch': {
                'db': os.path.join(work_Dir, *['out', f'T{protype}', 'data', f'scratch_t{protype}.json']),
                'acctag': {
                    't': {'p': f't{protype}_t_p_acc', 'n': f't{protype}_t_n_acc'},
                    'v': {'p': f't{protype}_v_p_acc', 'n': f't{protype}_v_n_acc'},
                },
                'ssatag': {
                    't': {'p': f't{protype}_t_p_ssa', 'n': f't{protype}_t_n_ssa'},
                    'v': {'p': f't{protype}_v_p_ssa', 'n': f't{protype}_v_n_ssa'},
                }
            },
        }
        for tag_style in ['acc', 'ssa']:
            for pn in ['p', 'n']:
                path_to_out = os.path.join(
                    work_Dir, *['tmp', "scratchfasta", f"T{protype}", f"{tag_style}", ])
                if os.path.exists(path_to_out) == False:
                    os.makedirs(path_to_out)
                extract_from_files(
                    path_to_json=config['scratch']['db'],
                    path_to_out=os.path.join(path_to_out, f"{pn}.fasta"),
                    tagname_list=[
                        config['scratch'][f'{tag_style}tag']['t'][pn], config['scratch'][f'{tag_style}tag']['v'][pn]],
                    cterreverse=True if ter == 'c' else False
                )

# %%
