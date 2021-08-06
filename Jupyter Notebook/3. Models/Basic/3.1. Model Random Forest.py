from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.model_selection import KFold
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import auc

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.set_printoptions(suppress=True)

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.max_colwidth = None

from IPython.display import clear_output

import json

MODEL_FILE = "Random_Forest_All.json"

# Entscheidungsbaum
TREE_PARAMS = {
        "max_depth": 10,
        #"max_features" : [50, 100, 140, 'auto'],
        "max_samples" : None,
        "min_samples_leaf" : 50,
        "n_estimators": 1300,
        "n_jobs" : -1
}

path1 = Path(r"A:\Workspace\Python\Masterarbeit\Kaggle Home Credit Datensatz")
path2 = Path(r"C:\Users\rober\Documents\Workspace\Python\Masterarbeit\Kaggle Home Credit Datensatz")

if path1.is_dir():
    DATASET_DIR = path1
else:
    DATASET_DIR = path2

model_path = DATASET_DIR / "Models" / MODEL_FILE
MODEL_EXIST = model_path.is_file()

app_train_mets = pd.read_csv(DATASET_DIR / "Datenaufbereitung" / "app_train_mets.csv", index_col="SK_ID_CURR")
app_train_cats = pd.read_csv(DATASET_DIR / "Datenaufbereitung" / "app_train_cats.csv", index_col="SK_ID_CURR")
bureau_mets = pd.read_csv(DATASET_DIR / "Datenaufbereitung" / "bureau_mets.csv", index_col="SK_ID_CURR")
bureau_cats = pd.read_csv(DATASET_DIR / "Datenaufbereitung" / "bureau_cats.csv", index_col="SK_ID_CURR")
pa_mets = pd.read_csv(DATASET_DIR / "Datenaufbereitung" / "pa_mets.csv", index_col="SK_ID_CURR")
pa_cats = pd.read_csv(DATASET_DIR / "Datenaufbereitung" / "pa_cats.csv", index_col="SK_ID_CURR")
ip_mets = pd.read_csv(DATASET_DIR / "Datenaufbereitung" / "ip_mets.csv", index_col="SK_ID_CURR")
pos_mets = pd.read_csv(DATASET_DIR / "Datenaufbereitung" / "pos_mets.csv", index_col="SK_ID_CURR")

app_train = pd.merge(app_train_mets, app_train_cats, left_index=True, right_index=True)
app_train = pd.merge(app_train, bureau_mets, left_index=True, right_index=True)
app_train = pd.merge(app_train, bureau_cats, left_index=True, right_index=True)
app_train = pd.merge(app_train, pa_mets, left_index=True, right_index=True)
app_train = pd.merge(app_train, pa_cats, left_index=True, right_index=True)
app_train = pd.merge(app_train, ip_mets, left_index=True, right_index=True)
app_train = pd.merge(app_train, pos_mets, left_index=True, right_index=True)

y = app_train["TARGET"]
x = app_train.drop(["TARGET"], axis=1)

if not MODEL_EXIST:
    # unterteilt den trainingsdatensatz in trainings- und validierungsdatensätze
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5, random_state=0)

    # logistisches Regressionsmodell
    model = RandomForestClassifier(**TREE_PARAMS)
    model.fit(x_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    fpr, tpr, threshold = roc_curve(y_test, model.predict_proba(x_test)[:, 1])
    print(auc)

    # Koeffizienten der einzelnen Klassen
    coef_dict = {}
    for coef, feat in zip(model.feature_importances_, x.columns.values):
        coef_dict[feat] = coef

    # Feature Importance
    d = dict(sorted(coef_dict.items(), key=lambda item: item[1], reverse=True))
    order = list(d.keys())

    model_data = {
        "todo": order,
        "keep": [],
        "drop": [],
        "params": TREE_PARAMS,
        "auc": [],
        "p": [],
        "n": []
    }

    auc_temp = 0.5

if MODEL_EXIST:
    with open(DATASET_DIR / "Models" / MODEL_FILE, 'r') as file:
        model_data = json.load(file)
    auc_temp = model_data["auc"][-1]

df = app_train

for head in model_data["todo"]:
    print(auc_temp, len(model_data["todo"]), len(model_data["keep"]), len(model_data["drop"]))

    model_data["keep"].append(head)

    X = df[model_data["keep"] + ["TARGET"]]
    y = X["TARGET"]
    x = X.drop(["TARGET"], axis=1)

    model = RandomForestClassifier(**TREE_PARAMS).fit(x, y)

    aucs = []

    kfold = KFold(2, shuffle=True, random_state=1)

    # enumerate splits
    for (train, test) in kfold.split(x):
        model.fit(x.iloc[train], y.iloc[train])
        auc = roc_auc_score(y.iloc[test], model.predict_proba(x.iloc[test])[:, 1])
        aucs.append(auc)

    auc = np.mean(aucs)
    n = len(X)
    p = len(X.columns)

    if auc > auc_temp:

        model_data["auc"].append(auc)
        model_data["p"].append(p)
        model_data["n"].append(n)

        auc_temp = auc

    else:
        model_data["keep"].remove(head)
        model_data["drop"].append(head)

    model_data["todo"].remove(head)

    with open(DATASET_DIR / "Models" / MODEL_FILE, 'w') as file:
        json.dump(model_data, file)

    clear_output(wait=True)
