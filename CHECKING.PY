import pandas as pd

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

print("Train columns:", train.columns.tolist())
print("Test columns:", test.columns.tolist())
