
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# --------------------------------------------------
# 1. Project paths and dataset loading
# --------------------------------------------------

BASE_DIR = Path(__file__).parent

DATA_PATH = BASE_DIR / "dataset" / "image_features.csv"
MODEL_DIR = BASE_DIR / "models"
GRAPH_DIR = BASE_DIR / "graphs"

MODEL_DIR.mkdir(exist_ok=True)
GRAPH_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. Separate training and testing data
# --------------------------------------------------

train_df = df[df["split"] == "train"].copy()
test_df = df[df["split"] == "test"].copy()

# Remove non-feature columns
drop_columns = ["image_name", "split", "label"]

X = train_df.drop(columns=drop_columns)
y = train_df["label"]

X_test = test_df.drop(columns=drop_columns)
y_test = test_df["label"]


# --------------------------------------------------
# 3. Create training and validation sets
# --------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nData split completed!")
print("Training records:", X_train.shape[0])
print("Validation records:", X_val.shape[0])
print("Testing records:", X_test.shape[0])


# --------------------------------------------------
# 4. Define ML algorithms
# --------------------------------------------------

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=12,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
}


# --------------------------------------------------
# 5. Train and validate all models
# --------------------------------------------------

results = []

trained_models = {}

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)

    # Predict validation data
    val_predictions = model.predict(X_val)

    # Calculate validation metrics
    accuracy = accuracy_score(y_val, val_predictions)

    precision = precision_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    trained_models[model_name] = model

    print(f"Validation Accuracy:  {accuracy:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall:    {recall:.4f}")
    print(f"Validation F1 Score:  {f1:.4f}")


# --------------------------------------------------
# 6. Compare models and select the best
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n========== VALIDATION MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

results_df.to_csv(
    GRAPH_DIR / "validation_model_comparison.csv",
    index=False
)

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\nBest model:", best_model_name)


# --------------------------------------------------
# 7. Final evaluation using untouched test data
# --------------------------------------------------

print("\n========== FINAL TEST EVALUATION ==========")

test_predictions = best_model.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test, test_predictions))

print("AI-generated Precision:",
      precision_score(y_test, test_predictions))

print("AI-generated Recall:",
      recall_score(y_test, test_predictions))

print("AI-generated F1 Score:",
      f1_score(y_test, test_predictions))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["Real", "AI-generated"]
    )
)


# --------------------------------------------------
# 8. Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, test_predictions)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Real", "AI-generated"],
    yticklabels=["Real", "AI-generated"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title(f"Confusion Matrix - {best_model_name}")

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "final_confusion_matrix.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 9. Validation model comparison graph
# --------------------------------------------------

plt.figure(figsize=(9, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="F1 Score"
)

plt.title("Validation F1 Score Comparison")
plt.ylim(0, 1)
plt.ylabel("F1 Score")
plt.xlabel("ML Algorithms")

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "model_comparison.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 10. Save the best model
# --------------------------------------------------

joblib.dump(
    best_model,
    MODEL_DIR / "best_model.joblib"
)

print("\nBest model saved successfully!")

print("Model:", best_model_name)
print("Location:", MODEL_DIR / "best_model.joblib")

print("\nModel training and evaluation completed!")
