import torch
import json
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


def train_uncertainty_probe():
    # 1. Load the vectors we generated
    if not os.path.exists("data/steering_vectors.json"):
        print("Error: data/steering_vectors.json not found! Run compute_null_projections.py first.")
        return

    with open("data/steering_vectors.json", "r") as f:
        data = json.load(f)

    # Extract features (X) and labels (y)
    X = np.array([item["vector"] for item in data])
    y = np.array([item["label"] for item in data])

    print(f"Loaded {len(X)} vectors for classification.")

    # 2. Split into training and testing sets (80-20 split)
    # stratify=y ensures equal proportions of Epistemic/Aleatoric in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Train a Linear Probe (Logistic Regression)
    # We use a simple linear model to prove the separation is clear in the vector space
    probe = LogisticRegression(max_iter=1000)
    probe.fit(X_train, y_train)

    # 4. Evaluation
    y_pred = probe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 40)
    print("LINEAR PROBE RESULTS")
    print("=" * 40)
    print(f"Overall Accuracy: {accuracy:.4%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Epistemic", "Aleatoric"]))

    # 5. Visualization: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Epistemic", "Aleatoric"],
                yticklabels=["Epistemic", "Aleatoric"])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Uncertainty Type Classification\nAccuracy: {accuracy:.2f}')

    # Save results
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/confusion_matrix.png")

    print("\n[✓] Results saved:")
    print(" - Confusion Matrix: results/confusion_matrix.png")
    print("=" * 40)

    # 6. Bonus: Check feature importance (optional)
    # This shows which of the 12 Null-Space dimensions are most important
    importance = np.abs(probe.coef_[0])
    print(f"Top 3 influential Null-Space dimensions: {np.argsort(importance)[-3:][::-1]}")


if __name__ == "__main__":
    train_uncertainty_probe()