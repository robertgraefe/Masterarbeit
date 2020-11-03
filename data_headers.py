from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).parent.absolute()
DATASET_DIR = Path(r"C:\Users\rober\Google Drive\Universität\Masterarbeit\Datensatz\2. Datasets Kaggle")

tables = ["application_test", "application_train", "bureau", "bureau_balance", "credit_card_balance",
          "HomeCredit_columns_description", "installments_payments", "POS_CASH_balance", "previous_application",
          "sample_submission"]

pd.options.display.max_rows = None
pd.options.display.max_columns = None

for table in tables:
    try:
        url = DATASET_DIR / str(table + ".csv")
        dataframe = pd.read_csv(url)
        head = dataframe.head()
        head.to_html(BASE_DIR / "heads" / str(table + ".html"))
    except Exception as e:
        print(table)
        print(e)