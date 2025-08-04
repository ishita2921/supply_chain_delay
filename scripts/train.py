import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from preprocess import load_and_split_data, scale_data

def train_and_save_model():
    X_train, X_test, y_train, y_test = load_and_split_data(r'C:/Users/HP/PCOD-DETECTION/data/processed/Day5-preprocessed.csv')
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # ✅ Create models folder if it doesn't exist
    os.makedirs('../models', exist_ok=True)

    # ✅ Save model and scaler
    joblib.dump(model, '../models/best_model.pkl')
    joblib.dump(scaler, '../models/scaler.pkl')

if __name__ == '__main__':
    train_and_save_model()
