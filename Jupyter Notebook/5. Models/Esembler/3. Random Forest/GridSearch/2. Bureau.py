from sklearn.model_selection import GridSearchCV
from pathlib import Path
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier

path1 = Path(r"A:\Workspace\Python\Masterarbeit\Kaggle Home Credit Datensatz")
path2 = Path(r"C:\Users\rober\Documents\Workspace\Python\Masterarbeit\Kaggle Home Credit Datensatz")

if path1.is_dir():
    DATASET_DIR = path1
else:
    DATASET_DIR = path2

app_train = pd.read_csv(DATASET_DIR / "4. FillNA" / "application.csv")
bureau = pd.read_csv(DATASET_DIR / "4. FillNA" / "bureau.csv")
pa_pos = pd.read_csv(DATASET_DIR / "4. FillNA" / "pa_pos.csv")
pa_ip = pd.read_csv(DATASET_DIR / "4. FillNA" / "pa_ip.csv")
pa_cc = pd.read_csv(DATASET_DIR / "4. FillNA" / "pa_cc.csv")

app_train = app_train.set_index("SK_ID_CURR")
bureau = bureau.set_index("SK_ID_CURR")
pa_pos = pa_pos.set_index("SK_ID_CURR")
pa_ip = pa_ip.set_index("SK_ID_CURR")
pa_cc = pa_cc.set_index("SK_ID_CURR")

MODEL_BUREAU = "3.2. Esembler_RandomForest_bureau.json"
MODEL_PA_POS = "3.3. Esembler_RandomForest_pa_pos.json"
MODEL_PA_IP = "3.4. Esembler_RandomForest_pa_ip.json"
MODEL_PA_CC = "3.5. Esembler_RandomForest_pa_cc.json"

with open(DATASET_DIR / "Models" / MODEL_BUREAU, 'r') as file:
    model_bureau_data = json.load(file)

with open(DATASET_DIR / "Models" / MODEL_PA_POS, 'r') as file:
    model_pa_pos_data = json.load(file)

with open(DATASET_DIR / "Models" / MODEL_PA_IP, 'r') as file:
    model_pa_ip_data = json.load(file)

with open(DATASET_DIR / "Models" / MODEL_PA_CC, 'r') as file:
    model_pa_cc_data = json.load(file)

# Random Forest
TREE_PARAMS = {
    "max_depth": [15],
    "min_samples_leaf": [15,17,20],
#    "n_estimators": [800, 900, 1000],
    "random_state": [0],
    "n_jobs": [-1]
}

model = RandomForestClassifier()

gridsearch = GridSearchCV(model, TREE_PARAMS, scoring='roc_auc', n_jobs=-1)

x = bureau[model_bureau_data["keep"]]
print(len(x.head()))
y = app_train.loc[bureau.index]["TARGET"]

gridsearch.fit(x,y)

print(gridsearch.best_params_)
print(gridsearch.best_score_)

model_bureau_data["params"] = gridsearch.best_params_

with open(DATASET_DIR / "Models" / MODEL_BUREAU, 'w') as file:
    json.dump(model_bureau_data, file)