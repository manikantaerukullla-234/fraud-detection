import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
from preprocess import load_data, preprocess


def train_model(X_train, y_train):
    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    print("Training complete.")
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n===== Classification Report =====")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    roc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {roc:.4f}")

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc:.3f}", color="steelblue")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Fraud Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=150)
    print("ROC curve saved to roc_curve.png")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")
    return roc


def save_model(model, path="model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    df = load_data("data/creditcard.csv")
    X_train, X_test, y_train, y_test = preprocess(df)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)
