from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent.absolute()
DATASET_DIR = Path(r"C:\Users\rober\Google Drive\Universität\Masterarbeit\Datensatz\2. Datasets Kaggle")

tables = ["application_test", "application_train", "bureau", "bureau_balance", "credit_card_balance",
          "HomeCredit_columns_description", "installments_payments", "POS_CASH_balance", "previous_application",
          "sample_submission"]

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
