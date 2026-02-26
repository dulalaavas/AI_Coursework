import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==============================
# 1. LOAD DATASET (SEMICOLON)
# ==============================

df = pd.read_csv("student-mat.csv", sep=";")

print("Raw shape:", df.shape)

# ==============================
# 2. FIX DATA TYPES
# ==============================

# Convert grade columns to numeric (important because some are strings)
df["G1"] = pd.to_numeric(df["G1"])
df["G2"] = pd.to_numeric(df["G2"])
df["G3"] = pd.to_numeric(df["G3"])

# ==============================
# 3. CREATE TARGET VARIABLE
# ==============================

df["pass"] = np.where(df["G3"] >= 10, 1, 0)

# Drop grade columns
df = df.drop(["G1", "G2", "G3"], axis=1)

X = df.drop("pass", axis=1)
y = df["pass"]

print("Processed shape:", df.shape)
print("\nClass distribution:")
print(y.value_counts())

# ==============================
# 4. ENCODE CATEGORICAL FEATURES
# ==============================

X = pd.get_dummies(X, drop_first=True)

# ==============================
# 5. TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 6. DECISION TREE
# ==============================

tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)

print("\n--- Decision Tree ---")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print("Precision:", precision_score(y_test, y_pred_tree))
print("Recall:", recall_score(y_test, y_pred_tree))
print("F1 Score:", f1_score(y_test, y_pred_tree))

print("Training Accuracy:", tree_model.score(X_train, y_train))
print("Testing Accuracy:", tree_model.score(X_test, y_test))

# Limited depth to reduce overfitting
tree_limited = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_limited.fit(X_train, y_train)

print("Limited Depth Test Accuracy:", tree_limited.score(X_test, y_test))

# ==============================
# 7. CONFUSION MATRIX
# ==============================

print("\nConfusion Matrix (Decision Tree):")
print(confusion_matrix(y_test, y_pred_tree))

# ==============================
# 8. FEATURE IMPORTANCE
# ==============================

importances = tree_model.feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance_df.head(10))