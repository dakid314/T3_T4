'''
Author: George Zhao
Date: 2022-02-24 20:12:49
LastEditors: George Zhao
LastEditTime: 2022-02-24 21:36:21
Description: 
Email: 2018221138@email.szu.edu.cn
Company: SZU
Version: 1.0
'''
# %%
import os
import sys
import warnings
sys.path.append('src')

from Bio import SeqIO, Seq, SeqRecord
# %%

# Get Bigger 30.


def spilt_fasta_File(
    path_to_fasta: str,
    path_to_output_withPattern: str,
    MinLength=31,
    cter: bool = False
):

    non_T3SEs_list = list(SeqIO.parse(
        path_to_fasta, 'fasta'))

    for i in range(len(non_T3SEs_list)):
        if len(str(non_T3SEs_list[i].seq)) < MinLength:
            warnings.warn(
                f'Found Seq:{non_T3SEs_list[i].id} \'s Length: {len(str(non_T3SEs_list[i].seq))} <= 30')

            non_T3SEs_list[i] = SeqRecord.SeqRecord(
                Seq.Seq(
                    (
                        str(non_T3SEs_list[i].seq)
                        +
                        'A' * (MinLength - len(str(non_T3SEs_list[i].seq)))
                    )
                    if cter == False else
                    (
                        'A' * (MinLength - len(str(non_T3SEs_list[i].seq)))
                        +
                        str(non_T3SEs_list[i].seq)
                    )
                ),
                id=non_T3SEs_list[i].id,
                name=non_T3SEs_list[i].name,
                description=non_T3SEs_list[i].description
            )
        else:
            pass

    if os.path.exists(os.path.split(path_to_output_withPattern)[0]) == False:
        os.makedirs(os.path.split(path_to_output_withPattern)[0])

    with open(path_to_output_withPattern, 'w+', encoding='UTF-8') as f:
        SeqIO.write(
            non_T3SEs_list,
            f,
            'fasta'
        )


# %%
if __name__ == '__main__':
    import utils
    work_Dir = utils.workdir.workdir(os.getcwd(), 3)
    rdata_Dir = os.path.join(work_Dir, 'data')
    tmp_Dir = os.path.join(work_Dir, 'tmp')

    # T1
    spilt_fasta_File(
        path_to_fasta=os.path.join(
            rdata_Dir, *['db', 'T1', 't_n.fasta']),
        path_to_output_withPattern=os.path.join(
            tmp_Dir, *['Bigger30', 'T1', 't_n.bigger30.fasta']
        ),
        cter=True
    )
    spilt_fasta_File(
        path_to_fasta=os.path.join(
            rdata_Dir, *['db', 'T1', 'v_n.fasta']),
        path_to_output_withPattern=os.path.join(
            tmp_Dir, *['Bigger30', 'T1', 'v_n.bigger30.fasta']
        ),
        cter=True
    )
    spilt_fasta_File(
        path_to_fasta=os.path.join(
            rdata_Dir, *['db', 'T1', 't_p.fasta']),
        path_to_output_withPattern=os.path.join(
            tmp_Dir, *['Bigger30', 'T1', 't_p.bigger30.fasta']
        ),
        cter=True
    )
    spilt_fasta_File(
        path_to_fasta=os.path.join(
            rdata_Dir, *['db', 'T1', 'v_p.fasta']),
        path_to_output_withPattern=os.path.join(
            tmp_Dir, *['Bigger30', 'T1', 'v_p.bigger30.fasta']
        ),
        cter=True
    )

    # T2
    spilt_fasta_File(
        path_to_fasta=os.path.join(
            rdata_Dir, *['db', 'T2', 't_n.fasta']),
        path_to_output_withPattern=os.path.join(
            tmp_Dir, *['Bigger30', 'T2', 't_n.bigger30.fasta']
        )
    )
    spilt_fasta_File(
        path_to_fasta=os.path.join(
            rdata_Dir, *['db', 'T2', 'v_n.fasta']),
        path_to_output_withPattern=os.path.join(
            tmp_Dir, *['Bigger30', 'T2', 'v_n.bigger30.fasta']
        )
    )
    spilt_fasta_File(
        path_to_fasta=os.path.join(
            rdata_Dir, *['db', 'T2', 't_p.fasta']),
        path_to_output_withPattern=os.path.join(
            tmp_Dir, *['Bigger30', 'T2', 't_p.bigger30.fasta']
        )
    )
    spilt_fasta_File(
        path_to_fasta=os.path.join(
            rdata_Dir, *['db', 'T2', 'v_p.fasta']),
        path_to_output_withPattern=os.path.join(
            tmp_Dir, *['Bigger30', 'T2', 'v_p.bigger30.fasta']
        )
    )

    # T3
    spilt_fasta_File(
        path_to_fasta=os.path.join(
            rdata_Dir, *['db', 'T3', 't_n.fasta']),
        path_to_output_withPattern=os.path.join(
            tmp_Dir, *['Bigger30', 'T3', 't_n.bigger30.fasta']
        )
    )
    spilt_fasta_File(
        path_to_fasta=os.path.join(
            rdata_Dir, *['db', 'T3', 'v_n.fasta']),
        path_to_output_withPattern=os.path.join(
            tmp_Dir, *['Bigger30', 'T3', 'v_n.bigger30.fasta']
        )
    )
    spilt_fasta_File(
        path_to_fasta=os.path.join(
            rdata_Dir, *['db', 'T3', 't_p.fasta']),
        path_to_output_withPattern=os.path.join(
            tmp_Dir, *['Bigger30', 'T3', 't_p.bigger30.fasta']
        )
    )
    spilt_fasta_File(
        path_to_fasta=os.path.join(
            rdata_Dir, *['db', 'T3', 'v_p.fasta']),
        path_to_output_withPattern=os.path.join(
            tmp_Dir, *['Bigger30', 'T3', 'v_p.bigger30.fasta']
        )
    )

    # T4
    spilt_fasta_File(
        path_to_fasta=os.path.join(
            rdata_Dir, *['db', 'T4', 't_n.fasta']),
        path_to_output_withPattern=os.path.join(
            tmp_Dir, *['Bigger30', 'T4', 't_n.bigger30.fasta']
        ),
        cter=True
    )
    spilt_fasta_File(
        path_to_fasta=os.path.join(
            rdata_Dir, *['db', 'T4', 'v_n.fasta']),
        path_to_output_withPattern=os.path.join(
            tmp_Dir, *['Bigger30', 'T4', 'v_n.bigger30.fasta']
        ),
        cter=True
    )
    spilt_fasta_File(
        path_to_fasta=os.path.join(
            rdata_Dir, *['db', 'T4', 't_p.fasta']),
        path_to_output_withPattern=os.path.join(
            tmp_Dir, *['Bigger30', 'T4', 't_p.bigger30.fasta']
        ),
        cter=True
    )
    spilt_fasta_File(
        path_to_fasta=os.path.join(
            rdata_Dir, *['db', 'T4', 'v_p.fasta']),
        path_to_output_withPattern=os.path.join(
            tmp_Dir, *['Bigger30', 'T4', 'v_p.bigger30.fasta']
        ),
        cter=True
    )

    # T6
    spilt_fasta_File(
        path_to_fasta=os.path.join(
            rdata_Dir, *['db', 'T6', 't_n.fasta']),
        path_to_output_withPattern=os.path.join(
            tmp_Dir, *['Bigger30', 'T6', 't_n.bigger30.fasta']
        )
    )
    spilt_fasta_File(
        path_to_fasta=os.path.join(
            rdata_Dir, *['db', 'T6', 'v_n.fasta']),
        path_to_output_withPattern=os.path.join(
            tmp_Dir, *['Bigger30', 'T6', 'v_n.bigger30.fasta']
        )
    )
    spilt_fasta_File(
        path_to_fasta=os.path.join(
            rdata_Dir, *['db', 'T6', 't_p.fasta']),
        path_to_output_withPattern=os.path.join(
            tmp_Dir, *['Bigger30', 'T6', 't_p.bigger30.fasta']
        )
    )
    spilt_fasta_File(
        path_to_fasta=os.path.join(
            rdata_Dir, *['db', 'T6', 'v_p.fasta']),
        path_to_output_withPattern=os.path.join(
            tmp_Dir, *['Bigger30', 'T6', 'v_p.bigger30.fasta']
        )
    )
