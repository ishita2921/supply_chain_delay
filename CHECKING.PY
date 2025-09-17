# import pandas as pd

# train = pd.read_csv("data/processed/train.csv")
# test = pd.read_csv("data/processed/test.csv")

# print("Train columns:", train.columns.tolist())
# print("Test columns:", test.columns.tolist())
import pandas as pd
from src.supply_chain import utils

df = pd.read_csv("data/processed/test.csv")
df = utils.standardize_columns(df)

numeric_df = df.select_dtypes(include=["number"])
print(numeric_df.corr()["delay_flag"].sort_values(ascending=False))
