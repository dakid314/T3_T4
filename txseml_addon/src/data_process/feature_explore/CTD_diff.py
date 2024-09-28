'''
Author: George Zhao
Date: 2022-05-20 19:53:03
LastEditors: George Zhao
LastEditTime: 2022-05-21 16:55:57
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
# %%
import os
import sys
sys.path.append("../..")
from libpybiofeature import featurebuilder
from libpyradarplot.radarplot import draw_radar

from Bio import SeqIO
import numpy as np
import pandas as pd
import scipy.stats as stats
# %%


def comparexCTD(path_to_p_fasta: str, path_to_n_fasta: str, desc: str, path_to_out_dir, pcutoff=0.05):

    if os.path.exists(path_to_out_dir) == False:
        os.makedirs(path_to_out_dir)

    pdb_ctdc_df = featurebuilder.build_CTDC_feature(
        path_to_fasta=path_to_p_fasta, seq_id_list=None
    )
    ndb_ctdc_df = featurebuilder.build_CTDC_feature(
        path_to_fasta=path_to_n_fasta, seq_id_list=None
    )
    pdb_df, ndb_df = pdb_ctdc_df, ndb_ctdc_df
    result = []
    for acname in pdb_df.columns:
        try:
            result.append(
                {
                    'keys': acname,
                    'P_Freqence_median': np.median(pdb_df.loc[:, acname]),
                    'N_Freqence_median': np.median(ndb_df.loc[:, acname]),
                    'pvalue': stats.mannwhitneyu(
                        pdb_df.loc[:, acname],
                        ndb_df.loc[:, acname],
                        alternative='two-sided'
                    ).pvalue
                }
            )
        except ValueError as e:
            # print(acname, e)
            result.append(
                {
                    'keys': acname,
                    'P_Freqence_median': np.median(pdb_df.loc[:, acname]),
                    'N_Freqence_median': np.median(ndb_df.loc[:, acname]),
                    'pvalue': 2
                }
            )
    result_df = pd.DataFrame(result)
    result_df['obv'] = result_df.apply(
        lambda r: "obv" if r['pvalue'] < pcutoff else "noobv",
        axis=1
    )

    with open(os.path.join(path_to_out_dir, f"{desc}_CTDC_stat.csv"), "w+", encoding='UTF-8') as f:
        f.write(f"# {desc}_CTDC_stat\n")
        result_df.to_csv(f, index=False)
    return result_df


# %%
if __name__ == "__main__":
    import utils
    work_Dir = utils.workdir.workdir(os.getcwd(), 4)
    for ty in [1, 2, 3, 4, 6]:
        comparexCTD(
            path_to_p_fasta=os.path.join(
                work_Dir, *['data', 'db', f'T{ty}', 'p.fasta']
            ),
            path_to_n_fasta=os.path.join(
                work_Dir, *['data', 'db', f'T{ty}', 'n.fasta']
            ),
            desc=f"T{ty}",
            path_to_out_dir=os.path.join(
                work_Dir, *['tmp', 'data_out_md_docs', 'research', f'T{ty}', ]
            )
        )

# %%
