import joblib

def predict_new(sample_features):
    # ✅ Corrected paths
    model = joblib.load('../models/best_model.pkl')
    scaler = joblib.load('../models/scaler.pkl')

    sample_scaled = scaler.transform([sample_features])
    prediction = model.predict(sample_scaled)
    return prediction[0]

# Test sample
if __name__ == '__main__':
    sample_input = [21.0, 162.0, 52.0, 80.0, 5.0, 13.0, 5.0, 1, 0, 1, 0, 1, 0, 1, 0]
    result = predict_new(sample_input)
    print("PCOS Prediction:", "Positive" if result == 1 else "Negative")
