# scripts/eval.py
import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from src.supply_chain import utils
from glob import glob

def find_data():
    files = glob("data/processed/*.parquet") + glob("data/interim/*.parquet")
    if files:
        return files[0]
    files = glob("data/processed/*.csv") + glob("data/interim/*.csv")
    if files:
        return files[0]
    raise FileNotFoundError("No data found for evaluation")

def main():
    # Load model + pipeline
    model = joblib.load("artifacts/best_model.pkl")
    pipeline = joblib.load("models/feature_pipeline.pkl")

    # Load validation data
    data_path = find_data()
    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    df = utils.standardize_columns(df)

    if "delay_flag" not in df.columns:
        raise RuntimeError("No target column (delay_flag) found in data")

    y = df["delay_flag"].astype(int)
    X = df.drop(columns=["delay_flag"])

    X_t = pipeline.transform(X)
    y_pred = model.predict(X_t)
    y_prob = model.predict_proba(X_t)[:, 1]

    print("🔎 Evaluation Results:")
    print(classification_report(y, y_pred))
    print("ROC AUC:", roc_auc_score(y, y_prob))

if __name__ == "__main__":
    main()
