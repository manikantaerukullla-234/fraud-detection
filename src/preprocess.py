import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
    return df


def preprocess(df: pd.DataFrame):
    # Scale Amount and Time (V1-V28 are already PCA-transformed)
    scaler = StandardScaler()
    df["scaled_amount"] = scaler.fit_transform(df[["Amount"]])
    df["scaled_time"] = scaler.fit_transform(df[["Time"]])
    df = df.drop(columns=["Amount", "Time"])

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Split before SMOTE to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply SMOTE only on training data
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    print(f"After SMOTE — training samples: {X_train_res.shape[0]}")
    print(f"Fraud in training: {y_train_res.sum()} | Non-fraud: {(y_train_res==0).sum()}")

    return X_train_res, X_test, y_train_res, y_test


if __name__ == "__main__":
    df = load_data("data/creditcard.csv")
    X_train, X_test, y_train, y_test = preprocess(df)
    print("Preprocessing complete.")
