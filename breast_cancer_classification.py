# Practical 4: Classification Models on the Breast Cancer Wisconsin Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, log_loss, confusion_matrix,
                             ConfusionMatrixDisplay)

# 1. Load the dataset (can also be loaded via pd.read_csv("breast_cancer_dataset.csv"))
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
class_names = cancer.target_names

# Save dataset to CSV for reference
df = pd.DataFrame(X, columns=cancer.feature_names)
df["target"] = y
df.to_csv("breast_cancer_dataset.csv", index=False)

print("Total samples:", X.shape[0])
print("Total features:", X.shape[1])
print("Classes:", list(class_names))

# 2. Split into training (80%) and testing (20%) data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 3. Create the models (SVM and KNN need feature scaling)
models = {
    "SVM": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Naive Bayes": GaussianNB()
}

# 4. Train each model and calculate the metrics
results = []
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro"),
        "Recall": recall_score(y_test, y_pred, average="macro"),
        "F1 Score": f1_score(y_test, y_pred, average="macro"),
        "Log Loss": log_loss(y_test, y_prob)
    })
    predictions[name] = y_pred

# 5. Show the comparison table
results_df = pd.DataFrame(results)
print("\nFINAL MODEL COMPARISON")
print(results_df.round(4).to_string(index=False))
results_df.to_csv("breast_cancer_model_comparison.csv", index=False)

# 6. Graph: accuracy, precision, recall and F1 score
results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]].plot(
    kind="bar", figsize=(10, 6)
)
plt.title("Breast Cancer Classification Model Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("screenshots/performance_comparison.png", dpi=300)
plt.show()

# 7. Graph: log loss (lower is better)
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["Log Loss"])
plt.title("Log Loss Comparison")
plt.ylabel("Log Loss (lower is better)")
plt.tight_layout()
plt.savefig("screenshots/log_loss_comparison.png", dpi=300)
plt.show()

# 8. Confusion matrix for every model
for name, y_pred in predictions.items():
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot()
    plt.title("Confusion Matrix - " + name)
    plt.tight_layout()
    plt.savefig("screenshots/confusion_matrix_" + name.replace(" ", "_") + ".png", dpi=300)
    plt.show()

# 9. Summary
print("\nBest accuracy:", results_df.loc[results_df["Accuracy"].idxmax(), "Model"])
print("Best F1 score:", results_df.loc[results_df["F1 Score"].idxmax(), "Model"])
print("Lowest log loss:", results_df.loc[results_df["Log Loss"].idxmin(), "Model"])
