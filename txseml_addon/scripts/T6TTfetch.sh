# PC-PSEAac
python3 -u /mnt/md0/Tools/PseinOne/src data/db/T6/t_p.fasta out/T6/data/tmp/PC-t_p.csv Protein PC-PseAAC -f csv -lamada 1 -w 0.5
python3 -u /mnt/md0/Tools/PseinOne/src data/db/T6/t_n.fasta out/T6/data/tmp/PC-t_n.csv Protein PC-PseAAC -f csv -lamada 1 -w 0.5
python3 -u /mnt/md0/Tools/PseinOne/src data/db/T6/v_p.fasta out/T6/data/tmp/PC-v_p.csv Protein PC-PseAAC -f csv -lamada 1 -w 0.5
python3 -u /mnt/md0/Tools/PseinOne/src data/db/T6/v_n.fasta out/T6/data/tmp/PC-v_n.csv Protein PC-PseAAC -f csv -lamada 1 -w 0.5
# SC-PSEAac
python3 -u /mnt/md0/Tools/PseinOne/src data/db/T6/t_p.fasta out/T6/data/tmp/SC-t_p.csv Protein SC-PseAAC -f csv -lamada 1 -w 0.5
python3 -u /mnt/md0/Tools/PseinOne/src data/db/T6/t_n.fasta out/T6/data/tmp/SC-t_n.csv Protein SC-PseAAC -f csv -lamada 1 -w 0.5
python3 -u /mnt/md0/Tools/PseinOne/src data/db/T6/v_p.fasta out/T6/data/tmp/SC-v_p.csv Protein SC-PseAAC -f csv -lamada 1 -w 0.5
python3 -u /mnt/md0/Tools/PseinOne/src data/db/T6/v_n.fasta out/T6/data/tmp/SC-v_n.csv Protein SC-PseAAC -f csv -lamada 1 -w 0.5

# Top_n_gram
python3 -u src/libexec/Top_n_gram_data_submiter.py -f data/db/T6/t_p.fasta -o out/T6/data/tmp/Topm-t_p.json
python3 -u src/libexec/Top_n_gram_data_submiter.py -f data/db/T6/t_n.fasta -o out/T6/data/tmp/Topm-t_n.json
python3 -u src/libexec/Top_n_gram_data_submiter.py -f data/db/T6/v_p.fasta -o out/T6/data/tmp/Topm-v_p.json
python3 -u src/libexec/Top_n_gram_data_submiter.py -f data/db/T6/v_n.fasta -o out/T6/data/tmp/Topm-v_n.json
