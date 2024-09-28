
python3 -u /mnt/md0/Public/T3_T4/txseml_addon/src/libexec/18pp.py -i "$new_filename" -o out/libfeatureselection/"$prot_type"/val_data/"$file_name"_18pp.csv 
python3 -u /mnt/md0/Public/T3_T4/txseml_addon/src/libexec/DPC.py -i "$new_filename" -o out/libfeatureselection/"$prot_type"/val_data/"$file_name"_DPC.csv 
python3 -u /mnt/md0/Public/T3_T4/txseml_addon/src/libexec/val_BPBaac.py -v "$new_filename" -o out/libfeatureselection/"$prot_type"/val_data/"$file_name"_BPBaac.csv -n /mnt/md0/Public/T3_T4/data/new_"$prot_type"/30/0/all_n"$prot_type"_30_1_1.fasta -p /mnt/md0/Public/T3_T4/data/new_"$prot_type"/pos/"$prot_type"_training_30.fasta -c $cter
python3 -u /mnt/md0/Public/T3_T4/txseml_addon/src/libexec/CTDC.py -i "$new_filename" -o out/libfeatureselection/"$prot_type"/val_data/"$file_name"_CTDC.csv 
python3 -u /mnt/md0/Public/T3_T4/txseml_addon/src/libexec/CTDD.py -i "$new_filename" -o out/libfeatureselection/"$prot_type"/val_data/"$file_name"_CTDD.csv 
python3 -u /mnt/md0/Public/T3_T4/txseml_addon/src/libexec/CTDT.py -i "$new_filename" -o out/libfeatureselection/"$prot_type"/val_data/"$file_name"_CTDT.csv 

python3 -u /mnt/md0/Public/T3_T4/txseml_addon/src/libexec/ctriad.py -i "$new_filename" -o out/libfeatureselection/"$prot_type"/val_data/"$file_name"_CTriad.csv 
python3 -u /mnt/md0/Public/T3_T4/txseml_addon/src/libexec/onehot.py -i "$new_filename" -o out/libfeatureselection/"$prot_type"/val_data/"$file_name"_onehot.csv -c $cter
python3 -u /mnt/md0/Public/T3_T4/txseml_addon/src/libexec/ppt.py -i "$new_filename" -o out/libfeatureselection/"$prot_type"/val_data/"$file_name"_ppt.csv 
python3 -u /mnt/md0/Public/T3_T4/txseml_addon/src/libexec/ppt25.py -i "$new_filename" -o out/libfeatureselection/"$prot_type"/val_data/"$file_name"_ppt25.csv -c $cter
python3 -u /mnt/md0/Tools/PseinOne/src "$new_filename" out/libfeatureselection/"$prot_type"/val_data/"$file_name"_SC-PseAAC.csv Protein SC-PseAAC -f csv -lamada 1 -w 0.5
python3 -u /mnt/md0/Tools/PseinOne/src "$new_filename" out/libfeatureselection/"$prot_type"/val_data/"$file_name"_PC-PseAAC.csv Protein PC-PseAAC -f csv -lamada 1 -w 0.5

python3 -u /mnt/md0/Public/T3_T4/txseml_addon/src/libexec/qso.py -i "$new_filename" -o out/libfeatureselection/"$prot_type"/val_data/"$file_name"_QSO.csv 
python3 -u /mnt/md0/Public/T3_T4/txseml_addon/src/libexec/AAC.py -i "$new_filename" -o out/libfeatureselection/"$prot_type"/val_data/"$file_name"_AAC.csv 
python3 -u /mnt/md0/Public/T3_T4/txseml_addon/src/libexec/TAC.py -i "$new_filename" -o out/libfeatureselection/"$prot_type"/val_data/"$file_name"_TAC.csv 
         



