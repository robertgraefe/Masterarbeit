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

app_train = app_train.set_index("SK_ID_CURR")

MODEL_APPLICATION = "3.1. Esembler_RandomForest_Application.json"

with open(DATASET_DIR / "Models" / MODEL_APPLICATION, 'r') as file:
    model_application_data = json.load(file)

# Random Forest
TREE_PARAMS = {
    "max_depth": [15],
    "min_samples_leaf": [12],
    "n_estimators": [900],
    "random_state": [0],
    "n_jobs": [-1]
}

model = RandomForestClassifier()

gridsearch = GridSearchCV(model, TREE_PARAMS, scoring='roc_auc', n_jobs=-1)

x = app_train[model_application_data["keep"]]
y = app_train.loc[app_train.index]["TARGET"]

gridsearch.fit(x,y)

print(gridsearch.best_params_)
print(gridsearch.best_score_)

model_application_data["params"] = gridsearch.best_params_

with open(DATASET_DIR / "Models" / MODEL_APPLICATION, 'w') as file:
    json.dump(model_application_data, file)