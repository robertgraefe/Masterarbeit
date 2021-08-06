from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
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

from sklearn.model_selection import GridSearchCV

import json

path1 = Path(r"A:\Workspace\Python\Masterarbeit\Kaggle Home Credit Datensatz")
path2 = Path(r"C:\Users\rober\Documents\Workspace\Python\Masterarbeit\Kaggle Home Credit Datensatz")

if path1.is_dir():
    DATASET_DIR = path1
else:
    DATASET_DIR = path2

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



param_grid = [
    {
        "max_depth": [5, 7, 10, None],
        "max_features" : [90, 100, 110, 'auto'],
        "min_samples_leaf" : [2, 1],
    }
]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5, random_state=0)

scoring = {'AUC': 'roc_auc'}

clf = GridSearchCV(
    DecisionTreeClassifier(), param_grid, scoring="roc_auc", n_jobs=-1
)
clf.fit(x_train, y_train)

print(clf.best_params_)