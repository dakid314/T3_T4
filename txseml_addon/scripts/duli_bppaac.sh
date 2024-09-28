folder="/mnt/md0/Public/T3_T4/data/30"
for subdir in "$folder"/*/; do
    if [ -d "$subdir" ]; then
        folder_name=$(basename "$subdir")
        path="out/libfeatureselection/val_data/BPBaac/$folder_name"
        if [ ! -d "$path" ]; then
            mkdir -p "$path"
        fi
        for file in "$subdir"/*; do
            if [ -f "$file" ]; then
                new_filename="${file%/*}/t_p.fasta"  # 新的文件名
                file_name=$(basename "$file")

                original_filename="${file%/*}/"$file_name""  # 原始文件名
                mv "$file" "$new_filename"
                python3 -u /mnt/md0/Public/T3_T4/txseml_addon/src/libexec/val_BPBaac.py -n "$new_filename" -p /mnt/md0/Public/T3_T4/data/pos/T3_training_30.fasta -o out/libfeatureselection/val_data/BPBaac/"$folder_name"/new_Ralstonia_pseudosolanacearum_GMI1000_"$file_name"_BPBaac.csv -v /mnt/md0/Public/T3_T4/data/val_tofeature/new_Ralstonia_pseudosolanacearum_GMI1000.fasta
                mv "$new_filename" "$original_filename"
            fi
        done
    fi
done