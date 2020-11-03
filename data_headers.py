import pandas as pd
import numpy as np

DATASET_DIR = r"C:\Users\rober\Google Drive\Masterarbeit\Datensatz\2. Datasets Kaggle\\"
# url = DATASET_DIR + "application_test.csv"
url = DATASET_DIR + "bureau.csv"


pd.options.display.max_rows = None
pd.options.display.max_columns = None

dataframe = pd.read_csv(url)
print(dataframe.loc[0:5][0:5])
