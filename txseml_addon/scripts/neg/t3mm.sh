folder="/mnt/md0/Public/T3_T4/data/30"
for subdir in "$folder"/*/; do
    if [ -d "$subdir" ]; then
        folder_name=$(basename "$subdir")
        path="out/libfeatureselection/t3mm/$folder_name"
        if [ ! -d "$path" ]; then
            mkdir -p "$path"
        fi
        for file in "$subdir"/*; do
            if [ -f "$file" ]; then
                new_filename="${file%/*}/t_p.fasta"  # 新的文件名
                file_name=$(basename "$file")
                original_filename="${file%/*}/"$file_name""  # 原始文件名
                mv "$file" "$new_filename"
                python3 -u /mnt/md0/Public/T3_T4/txseml_addon/src/libexec/t3mm.py -ineg "$new_filename" -ip /mnt/md0/Public/T3_T4/data/pos/T3_training_30.fasta -o out/libfeatureselection/t3mm/"$folder_name"/"$file_name".csv 
                mv "$new_filename" "$original_filename"
            fi
        done
    fi
done