from Bio import SeqIO
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_curve, confusion_matrix, precision_score, accuracy_score, f1_score, matthews_corrcoef, recall_score,auc,precision_recall_curve
import os
feature_name_list = ['18pp','AAC','BPBaac','CTDC','CTDT','CTriad','onehot',
                'PC-PseAAC','ppt25','QSO','SC-PseAAC','CTDD','DPC']
data = {'feature': '','batch':'','bac_name':'','stand':'', 'model': '', 'rate':'','rocAUC': '', 'prAUC': '', 'MCC': '', 'F1': '', 
        'Precision': '', 'Accuracy': '', 'Sensitivity': '', 'Specificity': '', 'FPR': '', 'Recall': '','pro_cutoff':''}
df = pd.DataFrame(columns=data.keys())

rate = '1_100'

bac_name = ['Ralstonia_pseudosolanacearum_GMI1000','Salmonella_LT2','Coxiella_burnetii_RSA_331',
            'new_Pseudomonas_sp.MIS38','new_Burkholderia_mallei_ATCC_23344','val'][5]


bac_type = ['T3','T4','T1','T2','T5'][4]

cd_hit = [30,70][1]
fasta_file =f"/mnt/md0/Public/T3_T4/data/new_{bac_type}/val_data/pos/val.fasta"
protein_ids = []
for seq_record in SeqIO.parse(fasta_file, "fasta"):
    protein_id = seq_record.id
    protein_ids.append(protein_id)
for feature_name in feature_name_list:
    batch = 0
    while batch <5:
        for model_name in ["XGBClassifier", "GaussianNB", "GradientBoostingClassifier",   
                                "SVC","KNeighborsClassifier", 
                                "RandomForestClassifier"]:
            
            model_save_dir = f"/mnt/md0/Public/T3_T4/model/{bac_type}/{cd_hit}_model/{feature_name}/{rate}/{batch}"

            val_df = pd.read_csv(f'/mnt/md0/Public/T3_T4/txseml_addon/out/libfeatureselection/{bac_type}/val_data/{bac_name}.fasta_{feature_name}.csv')
            val_df1 = val_df.iloc[0:, 1:]
            
            target_list = val_df['protein_id']
            target = []
            for a in range(len(target_list)):
                if target_list[a] in protein_ids:
                    target.append(1)
                else:
                    target.append(0)
            feature = pd.DataFrame(val_df1)
            if feature_name == 'CTriad':
                feature_ = np.array([eval(row) for row in feature['CTriad']])
            else:
                feature_ = feature.astype("float").values
                target_ = np.reshape(target, (len(target), 1))
            predict_result_list = []

            
            model = pickle.load(open(f"{model_save_dir}/{model_name}.pkl", "br"))

            pred = model.predict_proba(feature_)[:, 1]
            
            def calculate_fpr(y_true, y_pred):
                tn, fp, fn, tp = confusion_matrix(y_true,y_pred).ravel()
                fpr = fp / (fp + tn)
                return fpr
            fpr, tpr, thresholds = roc_curve(target_, pred)
            
            
            best_one_optimal_idx = np.argmax(tpr - fpr)
            pro_cutoff = thresholds[best_one_optimal_idx]
            
            pred_l = [1 if i >= pro_cutoff else 0 for i in pred]
            #后面新增的计算prAUC
            confusion_matrix_1d = confusion_matrix(target_, pred_l).ravel()
            confusion_dict = {N: n for N, n in zip(['tn', 'fp', 'fn', 'tp'], list(
                confusion_matrix_1d * 2 / np.sum(confusion_matrix_1d)))}
            
            FPR = calculate_fpr(target_,pred_l)
            
            precision, recall, _ = precision_recall_curve(target_, pred)
            pr_auc = auc(recall, precision)
            Recall = recall_score(target_, pred_l)
            evaluation = {
                'feature': feature_name, 
                'batch':batch,

                
                'model': model_name,
                'rate':f'{rate}',
                "rocAUC": auc(fpr, tpr),
                "prAUC": pr_auc,
                "MCC": matthews_corrcoef(target_, pred_l),
                "F1": f1_score(target_, pred_l),
                "Precision": precision_score(target_, pred_l,zero_division=1),
                "Accuracy": accuracy_score(target_, pred_l),
                "Sensitivity": confusion_dict['tp'] / (confusion_dict['tp'] + confusion_dict['fn']),
                "Specificity": confusion_dict['tn'] / (confusion_dict['tn'] + confusion_dict['fp']),
                "FPR":FPR,
                "Recall":Recall,
                'pro_cutoff': pro_cutoff
            }

            df = pd.concat([df, pd.DataFrame(evaluation, index=[0])], ignore_index=True)
        batch+=1


df.to_excel(f'{bac_type}_{cd_hit}_duli.xlsx', index=False)