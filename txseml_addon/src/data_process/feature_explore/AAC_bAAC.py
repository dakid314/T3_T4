'''
Author: George Zhao
Date: 2022-05-20 19:44:55
LastEditors: George Zhao
LastEditTime: 2022-05-21 12:37:22
Description: 比较组成差异、画出雷达图; 使用Freqence Median
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


def comparexAC(path_to_p_fasta: str, path_to_n_fasta: str, desc: str, path_to_out_dir, pcutoff=0.05, NCF=' F', terlength: int = None):
    # Argment desc: No NCF require.

    if NCF != 'F' and terlength is None:
        raise ValueError("NCF != 'F' and terlength is None")
    if NCF == 'F':
        desc = desc + '_F'
    else:
        desc = desc + f'_{NCF}{terlength}'

    if os.path.exists(path_to_out_dir) == False:
        os.makedirs(path_to_out_dir)

    pdb_aac_df = featurebuilder.build_acc_feature(
        path_to_fasta=path_to_p_fasta, seq_id_list=None, NCF=NCF, terlength=terlength
    )
    ndb_aac_df = featurebuilder.build_acc_feature(
        path_to_fasta=path_to_n_fasta, seq_id_list=None, NCF=NCF, terlength=terlength
    )
    pdb_dac0_df = featurebuilder.build_dac_feature(
        path_to_fasta=path_to_p_fasta, seq_id_list=None, interval=0, NCF=NCF, terlength=terlength
    )
    ndb_dac0_df = featurebuilder.build_dac_feature(
        path_to_fasta=path_to_n_fasta, seq_id_list=None, interval=0, NCF=NCF, terlength=terlength
    )
    pdb_dac1_df = featurebuilder.build_dac_feature(
        path_to_fasta=path_to_p_fasta, seq_id_list=None, interval=1, NCF=NCF, terlength=terlength
    )
    ndb_dac1_df = featurebuilder.build_dac_feature(
        path_to_fasta=path_to_n_fasta, seq_id_list=None, interval=1, NCF=NCF, terlength=terlength
    )
    pdb_dac2_df = featurebuilder.build_dac_feature(
        path_to_fasta=path_to_p_fasta, seq_id_list=None, interval=2, NCF=NCF, terlength=terlength
    )
    ndb_dac2_df = featurebuilder.build_dac_feature(
        path_to_fasta=path_to_n_fasta, seq_id_list=None, interval=2, NCF=NCF, terlength=terlength
    )
    pdb_tac_df = featurebuilder.build_tac_feature(
        path_to_fasta=path_to_p_fasta, seq_id_list=None, NCF=NCF, terlength=terlength
    )
    ndb_tac_df = featurebuilder.build_tac_feature(
        path_to_fasta=path_to_n_fasta, seq_id_list=None, NCF=NCF, terlength=terlength
    )
    df_result_list = []
    for pdb_df, ndb_df in [
        [pdb_aac_df, ndb_aac_df],
        [pdb_dac0_df, ndb_dac0_df],
        [pdb_dac1_df, ndb_dac1_df],
        [pdb_dac2_df, ndb_dac2_df],
        [pdb_tac_df, ndb_tac_df],
    ]:
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
        df_result_list.append(result_df)

    draw_radar(
        path_to_out=os.path.join(path_to_out_dir, f"{desc}_AAC_radar.pdf"),
        df=pd.concat([pdb_aac_df.median(), ndb_aac_df.median()],
                     axis=1).values.T,
        df_legend_label=["P", "N"],
        axisname=pdb_aac_df.columns,
        title=f"{desc}_AAC_radar",
        axisrgrids=[0, 0.025, 0.05, 0.1, 0.125, 0.15, 0.2]
    )
    toreturnresultdf = pd.concat(df_result_list)
    with open(os.path.join(path_to_out_dir, f"{desc}_AC_stat.csv"), "w+", encoding='UTF-8') as f:
        f.write(f"# {desc}_AC_stat\n")
        toreturnresultdf.to_csv(f, index=False)
    return toreturnresultdf


# %%
if __name__ == "__main__":
    import utils
    work_Dir = utils.workdir.workdir(os.getcwd(), 4)
    for ty in [1, 2, 3, 4, 6]:
        comparexAC(
            path_to_p_fasta=os.path.join(
                work_Dir, *['data', 'db', f'T{ty}', 'p.fasta']
            ),
            path_to_n_fasta=os.path.join(
                work_Dir, *['data', 'db', f'T{ty}', 'n.fasta']
            ),
            desc=f"T{ty}",
            path_to_out_dir=os.path.join(
                work_Dir, *['tmp', 'data_out_md_docs', 'research', f'T{ty}', ]
            ),
            NCF="F"
        )
        comparexAC(
            path_to_p_fasta=os.path.join(
                work_Dir, *['data', 'db', f'T{ty}', 'p.fasta']
            ),
            path_to_n_fasta=os.path.join(
                work_Dir, *['data', 'db', f'T{ty}', 'n.fasta']
            ),
            desc=f"T{ty}",
            path_to_out_dir=os.path.join(
                work_Dir, *['tmp', 'data_out_md_docs', 'research', f'T{ty}', ]
            ),
            NCF="N",
            terlength=30
        )
        comparexAC(
            path_to_p_fasta=os.path.join(
                work_Dir, *['data', 'db', f'T{ty}', 'p.fasta']
            ),
            path_to_n_fasta=os.path.join(
                work_Dir, *['data', 'db', f'T{ty}', 'n.fasta']
            ),
            desc=f"T{ty}",
            path_to_out_dir=os.path.join(
                work_Dir, *['tmp', 'data_out_md_docs', 'research', f'T{ty}', ]
            ),
            NCF="N",
            terlength=50
        )
        comparexAC(
            path_to_p_fasta=os.path.join(
                work_Dir, *['data', 'db', f'T{ty}', 'p.fasta']
            ),
            path_to_n_fasta=os.path.join(
                work_Dir, *['data', 'db', f'T{ty}', 'n.fasta']
            ),
            desc=f"T{ty}",
            path_to_out_dir=os.path.join(
                work_Dir, *['tmp', 'data_out_md_docs', 'research', f'T{ty}', ]
            ),
            NCF="N",
            terlength=100
        )
        comparexAC(
            path_to_p_fasta=os.path.join(
                work_Dir, *['data', 'db', f'T{ty}', 'p.fasta']
            ),
            path_to_n_fasta=os.path.join(
                work_Dir, *['data', 'db', f'T{ty}', 'n.fasta']
            ),
            desc=f"T{ty}",
            path_to_out_dir=os.path.join(
                work_Dir, *['tmp', 'data_out_md_docs', 'research', f'T{ty}', ]
            ),
            NCF="C",
            terlength=30
        )
        comparexAC(
            path_to_p_fasta=os.path.join(
                work_Dir, *['data', 'db', f'T{ty}', 'p.fasta']
            ),
            path_to_n_fasta=os.path.join(
                work_Dir, *['data', 'db', f'T{ty}', 'n.fasta']
            ),
            desc=f"T{ty}",
            path_to_out_dir=os.path.join(
                work_Dir, *['tmp', 'data_out_md_docs', 'research', f'T{ty}', ]
            ),
            NCF="C",
            terlength=50
        )
        comparexAC(
            path_to_p_fasta=os.path.join(
                work_Dir, *['data', 'db', f'T{ty}', 'p.fasta']
            ),
            path_to_n_fasta=os.path.join(
                work_Dir, *['data', 'db', f'T{ty}', 'n.fasta']
            ),
            desc=f"T{ty}",
            path_to_out_dir=os.path.join(
                work_Dir, *['tmp', 'data_out_md_docs', 'research', f'T{ty}', ]
            ),
            NCF="C",
            terlength=100
        )
