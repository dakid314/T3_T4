import os 
feature = ['18pp','AAC','BPBaac','CTDC','CTDT','CTriad','onehot',
                'PC-PseAAC','ppt25','QSO','SC-PseAAC','CTDD','DPC']
for fe in feature:
    for a in range(5):
        path = f'/mnt/md0/Public/T3_T4/txseml_addon/out/libfeatureselection/T5/feature_research_neg/70/{fe}/{a}'
        os.makedirs(path, exist_ok=True)