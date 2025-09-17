# scripts/eval.py
import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from supply_chain import utils

def main():
    # Load model & pipeline
    model = joblib.load("artifacts/best_model.pkl")
    pipeline = joblib.load("models/feature_pipeline.pkl")

    # Load test data
    df = pd.read_csv("data/processed/test.csv")
    df = utils.standardize_columns(df)

    X = df.drop(columns=["delay_flag"])
    y = df["delay_flag"].astype(int)

    # Predict
    Xt = pipeline.transform(X)
    y_pred = model.predict(Xt)
    y_prob = model.predict_proba(Xt)[:, 1]

    print("🔎 Evaluation Results:")
    print(classification_report(y, y_pred, digits=3))
    print("ROC AUC:", roc_auc_score(y, y_prob))

if __name__ == "__main__":
    main()
