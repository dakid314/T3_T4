# %%
import pandas as pd
import numpy as np
import json
import os
# %%
prot_type = "6"
result_df = pd.read_excel(
    f"out/libfeatureselection/T{prot_type}/model/Onehot/searched_result.xlsx",
    f"T{prot_type}",
    header=[0, 1],
    index_col=[0, 1]
)

feature_col = np.array(result_df["Feature_Selected"].columns)[
    np.argmax(result_df["Feature_Selected"].values, axis=1)
]
model_col = result_df["Model_Information"]['Classifier_Name'].values
tt_mcc = result_df['Best_Performance']['mmc'].values
tt_auc = result_df['Best_Performance']['rocAUC'].values
cv_mcc = result_df['5FoldCV_Performance']['mmc'].values
cv_auc = result_df['5FoldCV_Performance']['rocAUC'].values

data_out = f"out/libfeatureselection/bubble_plot/T{prot_type}/data.json"
os.makedirs(os.path.dirname(data_out), exist_ok=True)

with open(
    data_out,
    "w+",
    encoding="UTF-8"
) as f:
    json.dump({
        "Feature_Name": feature_col.tolist(),
        "Model_Type": model_col.tolist(),
        "TT_MCC": tt_mcc.tolist(),
        "TT_rocAUC": tt_auc.tolist(),
        "CV_MCC": cv_mcc.tolist(),
        "CV_rocAUC": cv_auc.tolist(),
        "Title": f"T{prot_type} Model Performance of Single Feature",
        "ProtType": prot_type
    }, f)

# %%
