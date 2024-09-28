'''
Author: George Zhao
Date: 2022-05-21 16:28:14
LastEditors: George Zhao
LastEditTime: 2022-05-22 22:00:36
Description: 计算PSSM矩阵在位置上评分的Entropy、IG
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
# %%
import os
import sys
import itertools
sys.path.append("../..")

from Bio import SeqIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
from scipy.stats import entropy
import tqdm


def information_gain(members, split):
    '''
    Measures the reduction in entropy after the split  
    :param v: Pandas Series of the members
    :param split:
    :return:
    '''
    entropy_before = entropy(members.value_counts(normalize=True))
    split.name = 'split'
    members.name = 'members'
    grouped_distrib = members.groupby(split) \
        .value_counts(normalize=True) \
        .reset_index(name='count') \
        .pivot_table(index='split', columns='members', values='count').fillna(0)
    entropy_after = entropy(grouped_distrib, axis=1)
    entropy_after *= split.value_counts(sort=False, normalize=True)
    return entropy_before - entropy_after.sum()
    # members = pd.Series(['yellow', 'yellow', 'green', 'green', 'blue'])
    # split = pd.Series([0, 0, 1, 1, 0])
    # print(information_gain(members, split))

# %%


def go(
    path_to_p_fasta: str,
    path_to_n_fasta: str,
    desc: str,
    path_to_out_dir: str,
    NC: str = 'N',
    terlength: int = None
):
    # Argment desc: No NC require.

    # Original max Seq length
    max_p_seq_length = max([len(seq.seq)
                            for seq in SeqIO.parse(path_to_p_fasta, "fasta")])
    max_n_seq_length = max([len(seq.seq)
                            for seq in SeqIO.parse(path_to_n_fasta, "fasta")])
    # Can't overflow neither p nor n.
    max_seq_length = min(max_p_seq_length, max_n_seq_length)

    # P OR N?
    if terlength is None:
        desc = desc + f'_{NC}'
    else:
        desc = desc + f'_{NC}{terlength}'
        # use terlength
        max_seq_length = min(max_seq_length, terlength)

    if NC == 'N':
        p_ac_db = list([list(str(seq.seq).ljust(max_seq_length))
                        for seq in SeqIO.parse(path_to_p_fasta, "fasta")])
        n_ac_db = list([list(str(seq.seq).ljust(max_seq_length))
                        for seq in SeqIO.parse(path_to_n_fasta, "fasta")])

    else:
        p_ac_db = list([list(reversed(str(seq.seq).rjust(max_seq_length)))
                        for seq in SeqIO.parse(path_to_p_fasta, "fasta")])
        n_ac_db = list([list(reversed(str(seq.seq).rjust(max_seq_length)))
                        for seq in SeqIO.parse(path_to_n_fasta, "fasta")])

    # Make split
    split_list = list(itertools.chain(*[
        [1 for _ in p_ac_db], [0 for _ in n_ac_db]
    ]))
    # Make grossdb
    grossdb = list(itertools.chain(*[
        p_ac_db, n_ac_db
    ]))

    # Calculate the IG in each position.
    ig_list = [
        information_gain(
            members=pd.Series([
                seqlist[acindex]
                for seqlist in grossdb if seqlist[acindex] != ' '
            ]),
            split=pd.Series([
                split_list[index]
                for index, seqlist in enumerate(grossdb) if seqlist[acindex] != ' '
            ])
        )
        for acindex in range(max_seq_length)
    ]

    fig = plt.figure(figsize=(10.8, 5.4),)
    sns.lineplot(x=range(1, max_seq_length + 1),
                 y=ig_list)
    plt.title(f"{desc}_IG_plot")
    plt.xlabel("AC position")
    plt.ylabel("IG")
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_out_dir, f"{desc}_IG.pdf"))
    plt.close(fig)

    # Calculate the IE(bit)
    p_ie_list = [
        entropy(
            np.unique([
                seqlist[acindex]
                for seqlist in p_ac_db if seqlist[acindex] != ' '
            ], return_counts=True)[1],
            base=2
        )
        for acindex in range(max_seq_length)
    ]
    n_ie_list = [
        entropy(
            np.unique([
                seqlist[acindex]
                for seqlist in n_ac_db if seqlist[acindex] != ' '
            ], return_counts=True)[1],
            base=2
        )
        for acindex in range(max_seq_length)
    ]

    fig = plt.figure(figsize=(10.8, 5.4),)

    sns.lineplot(
        x='Position',
        y='IE',
        hue='Label',
        data=pd.concat([pd.DataFrame(
            {
                'Position': range(1, max_seq_length + 1),
                'IE': p_ie_list,
                'Label': 'P'
            }
        ), pd.DataFrame(
            {
                'Position': range(1, max_seq_length + 1),
                'IE': n_ie_list,
                'Label': 'N'
            }
        )])
    )
    plt.title(f"{desc}_IE_plot")
    plt.xlabel("AC position")
    plt.ylabel("IE")
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_out_dir, f"{desc}_IE.pdf"))
    plt.close(fig)

    return


# %%
if __name__ == "__main__":
    import utils
    work_Dir = utils.workdir.workdir(os.getcwd(), 4)
    for ty in [1, 2, 3, 4, 6]:
        for terlenght in [10, 30, 50, 100, 200, 300, 400, 1000, None]:
            for nc in ['N', 'C']:
                go(
                    path_to_p_fasta=os.path.join(
                        work_Dir, *['data', 'db', f'T{ty}', 'p.fasta']
                    ),
                    path_to_n_fasta=os.path.join(
                        work_Dir, *['data', 'db', f'T{ty}', 'n.fasta']
                    ),
                    desc=f"T{ty}",
                    path_to_out_dir=os.path.join(
                        work_Dir, *['tmp', 'data_out_md_docs',
                                    'research', f'T{ty}', ]
                    ),
                    NC=nc,
                    terlength=terlenght
                )

# %%
