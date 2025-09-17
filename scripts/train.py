# scripts/train.py
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from supply_chain import utils

def main():
    # Load train data
    df = pd.read_csv("data/processed/train.csv")
    df = utils.standardize_columns(df)

    X = df.drop(columns=["delay_flag"])
    y = df["delay_flag"].astype(int)

    # Load pipeline
    pipeline = joblib.load("models/feature_pipeline.pkl")

    # Train model
    Xt = pipeline.transform(X)
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(Xt, y)

    # Save model
    joblib.dump(model, "artifacts/best_model.pkl")
    print("✅ Model trained on train.csv and saved.")

if __name__ == "__main__":
    main()
