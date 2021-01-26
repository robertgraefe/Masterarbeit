import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.absolute()


def df_to_html(df, name):
    try:
        head = df.head()
        head.to_html(BASE_DIR / "heads" / str(name + ".html"))

        style = """
           <style>
             table {
             font-family: Arial, Helvetica, sans-serif;
             border-collapse: collapse;
             width: 100%;
           }


           table td, table th {
             border: 1px solid #ddd;
             padding: 8px;
           }

           table tr:nth-child(even){background-color: #f2f2f2;}

           table tr:hover {background-color: #ddd;}

           table th {
             padding-top: 12px;
             padding-bottom: 12px;
             text-align: left;
             background-color: #4BB2AE;
             color: white;
           }
           </style>"""

        with open(BASE_DIR / "heads" / str(name + ".html"), 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(style.rstrip('\r\n') + '\n' + content)

    except Exception as e:
        print(name + "\n" + e)


DATASET_DIR = Path(r"C:\Users\rober\Google Drive\Universität\Masterarbeit\Datensatz\2. Datasets Kaggle")

application_test = pd.read_csv(DATASET_DIR / "application_test.csv")
application_train = pd.read_csv(DATASET_DIR / "application_train.csv")
bureau = pd.read_csv(DATASET_DIR / "bureau.csv")
bureau_balance = pd.read_csv(DATASET_DIR / "bureau_balance.csv")
credit_card_balance = pd.read_csv(DATASET_DIR / "credit_card_balance.csv")
installments_payments = pd.read_csv(DATASET_DIR / "installments_payments.csv")
pos_cash_balance = pd.read_csv(DATASET_DIR / "POS_CASH_balance.csv")
previous_application = pd.read_csv(DATASET_DIR / "previous_application.csv")
sample_submission = pd.read_csv(DATASET_DIR / "sample_submission.csv")

Y = application_train["TARGET"]
train_X = application_train.drop(["TARGET"], axis=1)

test_id = application_test["SK_ID_CURR"]
test_X = application_test

data = pd.concat([train_X, test_X], axis=0)
#df_to_html(data, "data")
print(data.shape)
