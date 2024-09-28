'''
Author: George Zhao
Date: 2022-05-20 19:47:09
LastEditors: George Zhao
LastEditTime: 2022-05-21 12:48:32
Description: 蛋白的长度分布
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
# %%
import os
import sys
sys.path.append("../..")

from Bio import SeqIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
# %%


def prot_length_distribution(path_to_p_fasta: str, path_to_n_fasta: str, desc: str, path_to_out_dir,):

    if os.path.exists(path_to_out_dir) == False:
        os.makedirs(path_to_out_dir)

    fig, ax = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(7.2, 7.2),
    )

    p_length_list = [
        len(seq.seq)
        for seq in SeqIO.parse(path_to_p_fasta, "fasta")
    ]
    sns.distplot(p_length_list, ax=ax[0], color='r')
    ax[0].set_title(f"{desc}_P_Length_Distribution")
    ax[0].text(0.99, 0.95, f'Minimum: {np.min(p_length_list)}\nMedian: {np.median(p_length_list)}\nMaximum: {np.max(p_length_list)}', horizontalalignment='right',
               verticalalignment='top', transform=ax[0].transAxes, bbox={'facecolor': '#e5e5e5', 'alpha': 0.1, 'pad': 2})
    ax[0].set_xlabel("Sequence Length")
    ax[0].set_ylabel("Prob")

    n_length_list = [
        len(seq.seq)
        for seq in SeqIO.parse(path_to_n_fasta, "fasta")
    ]
    sns.distplot(n_length_list, ax=ax[1], color='b')
    ax[1].set_title(f"{desc}_N_Length_Distribution")
    ax[1].text(0.99, 0.95, f'Minimum: {np.min(n_length_list)}\nMedian: {np.median(n_length_list)}\nMaximum: {np.max(n_length_list)}', horizontalalignment='right',
               verticalalignment='top', transform=ax[1].transAxes, bbox={'facecolor': '#e5e5e5', 'alpha': 0.1, 'pad': 2})
    ax[1].set_xlabel("Sequence Length")
    ax[1].set_ylabel("Prob")

    plt.tight_layout()
    plt.savefig(os.path.join(path_to_out_dir, *
                             [f"{desc}_Length_Distribution.pdf", ]))
    plt.close(fig)


# %%
if __name__ == "__main__":
    import utils
    work_Dir = utils.workdir.workdir(os.getcwd(), 4)
    for ty in [1, 2, 3, 4, 6]:
        prot_length_distribution(
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
