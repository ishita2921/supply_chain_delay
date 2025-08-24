import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, PrecisionRecallDisplay
import matplotlib
matplotlib.use("Agg")  # non-GUI backend, plots saved as images
import matplotlib.pyplot as plt
import os

# Create folder for plots if it doesn't exist
os.makedirs("../plots", exist_ok=True)  # relative to your project folder


# ------------------ Functions ------------------
def train_logreg(X_train, y_train):
    categorical_cols = X_train.select_dtypes(include=["object"]).columns
    numeric_cols = X_train.select_dtypes(exclude=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(class_weight="balanced", solver="liblinear"))
    ])

    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, title="Model Evaluation"):
    y_pred = model.predict(X_test)
    print(f"\n=== {title} ===")
    print(classification_report(y_test, y_pred))
    
    # ROC-AUC (binary only)
    if len(y_test.unique()) == 2:
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
        print("ROC-AUC:", roc_auc)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"{title} - Confusion Matrix")
        plt.savefig(f"../plots/{title.replace(' ', '_')}_confusion_matrix.png")
        plt.close()

        # Precision-Recall curve
        y_scores = model.predict_proba(X_test)[:,1]
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)
        pr_display.plot()
        plt.title(f"{title} - Precision-Recall Curve")
        plt.savefig(f"../plots/{title.replace(' ', '_')}_precision_recall.png")
        plt.close()
    
    # Precision-Recall curve (binary)
    if len(y_test.unique()) == 2:
        y_scores = model.predict_proba(X_test)[:,1]
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)
        pr_display.plot()
        plt.title(f"{title} - Precision-Recall Curve")
        plt.show()

# ------------------ Main ------------------
if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv(r"C:/Users/HP/supply-chain-delay/data/interim/smart_logistics_cleaned.csv")
    
    # Features & target
    X = df.drop(columns=[
        "timestamp", "asset_id", "shipment_status",
        "logistics_delay_reason", "logistics_delay"
    ])
    y = df["logistics_delay"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # ------------------ Dummy Baseline ------------------
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    evaluate_model(dummy, X_test, y_test, title="Dummy Baseline")
    
    # ------------------ Logistic Regression Baseline ------------------
    logreg = train_logreg(X_train, y_train)
    evaluate_model(logreg, X_test, y_test, title="Logistic Regression Baseline")


    # ------------------ Logistic Regression Baseline ------------------
logreg = train_logreg(X_train, y_train)
evaluate_model(logreg, X_test, y_test, title="Logistic Regression Baseline")

# ------------------ Save the trained model ------------------
import joblib
import os

model_dir = "../models_saved"
os.makedirs(model_dir, exist_ok=True)  # create folder if it doesn't exist
model_path = os.path.join(model_dir, "logistic_regression_baseline.pkl")

joblib.dump(logreg, model_path)
print(f"Trained Logistic Regression model saved at: {model_path}")

