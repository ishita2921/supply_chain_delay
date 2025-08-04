# preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_split_data(csv_path):
    df = pd.read_csv(csv_path)

    X = df.drop('PCOS (Y/N)', axis=1)
    y = df['PCOS (Y/N)']

    return train_test_split(X, y, test_size=0.2, random_state=42)

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
