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
pa_cc = pd.read_csv(DATASET_DIR / "4. FillNA" / "pa_cc.csv")

app_train = app_train.set_index("SK_ID_CURR")
pa_cc = pa_cc.set_index("SK_ID_CURR")

MODEL_PA_CC = "2.5. Esembler_DecisionTree_pa_cc.json"

with open(DATASET_DIR / "Models" / MODEL_PA_CC, 'r') as file:
    model_pa_cc_data = json.load(file)

# Decision Tree
TREE_PARAMS = {
    "max_depth": [9],
    "min_samples_leaf": [14],
    "random_state": [0],
    "n_jobs": [-1]
}

model = RandomForestClassifier()

gridsearch = GridSearchCV(model, TREE_PARAMS, scoring='roc_auc', n_jobs=-1)

x = pa_cc[model_pa_cc_data["keep"]]
y = app_train.loc[pa_cc.index]["TARGET"]

gridsearch.fit(x,y)

print(gridsearch.best_params_)
print(gridsearch.best_score_)

model_pa_cc_data["params"] = gridsearch.best_params_

with open(DATASET_DIR / "Models" / MODEL_PA_CC, 'w') as file:
    json.dump(model_pa_cc_data, file)