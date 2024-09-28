python3 -u /mnt/md0/Tools/PseinOne/src "data/db/T1/t_p.fasta" "out/libfeatureselection/PCPseAAC/data/t_p.csv"  Protein PC-PseAAC -f csv -lamada 1 -w 0.5
python3 -u /mnt/md0/Tools/PseinOne/src "data/db/T1/t_n.fasta" "out/libfeatureselection/PCPseAAC/data/t_n.csv"  Protein PC-PseAAC -f csv -lamada 1 -w 0.5
python3 -u /mnt/md0/Tools/PseinOne/src "data/db/T1/v_p.fasta" "out/libfeatureselection/PCPseAAC/data/v_p.csv"  Protein PC-PseAAC -f csv -lamada 1 -w 0.5
python3 -u /mnt/md0/Tools/PseinOne/src "data/db/T1/v_n.fasta" "out/libfeatureselection/PCPseAAC/data/v_n.csv"  Protein PC-PseAAC -f csv -lamada 1 -w 0.5

python3 -u /mnt/md0/Tools/PseinOne/src "data/T1SE/RTX_filted_prot.fasta" "out/libfeatureselection/PCPseAAC/data/rtx.csv"  Protein PC-PseAAC -f csv -lamada 1 -w 0.5
python3 -u /mnt/md0/Tools/PseinOne/src "data/T1SE/non-RTX_filted_prot.fasta" "out/libfeatureselection/PCPseAAC/data/non-rtx.csv"  Protein PC-PseAAC -f csv -lamada 1 -w 0.5
python3 -u /mnt/md0/Tools/PseinOne/src "data/T1SE/noT1SE_GDB.fasta" "out/libfeatureselection/PCPseAAC/data/nonT1.csv"  Protein PC-PseAAC -f csv -lamada 1 -w 0.5