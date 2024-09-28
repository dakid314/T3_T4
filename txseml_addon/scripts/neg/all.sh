prot_type="T3"
cd_hit="70"
cter="False"
folder="/mnt/md0/Public/T3_T4/data/new_"$prot_type"/"$cd_hit""
for subdir in "$folder"/*/; do
    if [ -d "$subdir" ]; then
        folder_name=$(basename "$subdir")
        for file in "$subdir"/*; do
            if [ -f "$file" ]; then
                new_filename="${file%/*}/t_p.fasta"  # 新的文件名
                file_name=$(basename "$file")
                original_filename="${file%/*}/"$file_name""  # 原始文件名
                mv "$file" "$new_filename"
                
                # python3 -u /mnt/md0/Public/T3_T4/txseml_addon/src/libexec/18pp.py -i "$new_filename" -o out/libfeatureselection/"$prot_type"/feature_research_neg/"$cd_hit"/18pp/"$folder_name"/"$file_name".csv
                # python3 -u /mnt/md0/Public/T3_T4/txseml_addon/src/libexec/DPC.py -i "$new_filename" -o out/libfeatureselection/"$prot_type"/feature_research_neg/"$cd_hit"/DPC/"$folder_name"/"$file_name".csv 
                # python3 -u /mnt/md0/Public/T3_T4/txseml_addon/src/libexec/BPBAac.py -n "$new_filename" -o out/libfeatureselection/"$prot_type"/feature_research_neg/"$cd_hit"/BPBaac/"$folder_name"/"$file_name".csv  -p /mnt/md0/Public/T3_T4/data/new_"$prot_type"/pos/"$prot_type"_training_"$cd_hit".fasta -c $cter
                # python3 -u /mnt/md0/Public/T3_T4/txseml_addon/src/libexec/CTDC.py -i "$new_filename" -o out/libfeatureselection/"$prot_type"/feature_research_neg/"$cd_hit"/CTDC/"$folder_name"/"$file_name".csv
                # python3 -u /mnt/md0/Public/T3_T4/txseml_addon/src/libexec/CTDD.py -i "$new_filename" -o out/libfeatureselection/"$prot_type"/feature_research_neg/"$cd_hit"/CTDD/"$folder_name"/"$file_name".csv 
                # python3 -u /mnt/md0/Public/T3_T4/txseml_addon/src/libexec/CTDT.py -i "$new_filename" -o out/libfeatureselection/"$prot_type"/feature_research_neg/"$cd_hit"/CTDT/"$folder_name"/"$file_name".csv 
            
                # python3 -u /mnt/md0/Public/T3_T4/txseml_addon/src/libexec/ctriad.py -i "$new_filename" -o out/libfeatureselection/"$prot_type"/feature_research_neg/"$cd_hit"/CTriad/"$folder_name"/"$file_name".csv 
                # python3 -u /mnt/md0/Public/T3_T4/txseml_addon/src/libexec/onehot.py -i "$new_filename" -o out/libfeatureselection/"$prot_type"/feature_research_neg/"$cd_hit"/onehot/"$folder_name"/"$file_name".csv -c $cter
                python3 -u /mnt/md0/Public/T3_T4/txseml_addon/src/libexec/ppt.py -i "$new_filename" -o out/libfeatureselection/"$prot_type"/feature_research_neg/"$cd_hit"/ppt/"$folder_name"/"$file_name".csv 
                # python3 -u /mnt/md0/Public/T3_T4/txseml_addon/src/libexec/ppt25.py -i "$new_filename" -o out/libfeatureselection/"$prot_type"/feature_research_neg/"$cd_hit"/ppt25/"$folder_name"/"$file_name".csv -c $cter
                # python3 -u /mnt/md0/Tools/PseinOne/src "$new_filename" out/libfeatureselection/"$prot_type"/feature_research_neg/"$cd_hit"/SC-PseAAC/"$folder_name"/"$file_name".csv Protein SC-PseAAC -f csv -lamada 1 -w 0.5
                # python3 -u /mnt/md0/Tools/PseinOne/src "$new_filename" out/libfeatureselection/"$prot_type"/feature_research_neg/"$cd_hit"/PC-PseAAC/"$folder_name"/"$file_name".csv Protein PC-PseAAC -f csv -lamada 1 -w 0.5
            
                # python3 -u /mnt/md0/Public/T3_T4/txseml_addon/src/libexec/qso.py -i "$new_filename" -o out/libfeatureselection/"$prot_type"/feature_research_neg/"$cd_hit"/QSO/"$folder_name"/"$file_name".csv 

                # python3 -u /mnt/md0/Public/T3_T4/txseml_addon/src/libexec/AAC.py -i "$new_filename" -o out/libfeatureselection/"$prot_type"/feature_research_neg/"$cd_hit"/AAC/"$folder_name"/"$file_name".csv  
                python3 -u /mnt/md0/Public/T3_T4/txseml_addon/src/libexec/TAC.py -i "$new_filename" -o out/libfeatureselection/"$prot_type"/feature_research_neg/"$cd_hit"/TAC/"$folder_name"/"$file_name".csv  
                mv "$new_filename" "$original_filename"
            fi
        done
    fi
done




