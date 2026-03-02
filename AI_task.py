import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import (
    train_test_split, GridSearchCV,
    StratifiedKFold, cross_val_score, learning_curve
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# ==============================
# 1. LOAD DATASET
# ==============================

df = pd.read_csv("student-mat.csv", sep=";")
print("Raw shape:", df.shape)

# ==============================
# 2. DATA CLEANING
# ==============================

# Fix numeric types
df["G1"] = pd.to_numeric(df["G1"], errors='coerce')
df["G2"] = pd.to_numeric(df["G2"], errors='coerce')
df["G3"] = pd.to_numeric(df["G3"], errors='coerce')

# Check for missing values before cleaning
print("\nMissing values before cleaning:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "None")
print("Total missing:", df.isnull().sum().sum())

# Remove duplicates
n_before = len(df)
df = df.drop_duplicates()
print(f"\nDuplicates removed: {n_before - len(df)}")

# Drop rows with missing values in key columns
df = df.dropna(subset=["G1", "G2", "G3"])

# Remove outliers in numeric columns using IQR (conservative: 3*IQR)
numeric_cols = ["age", "absences", "G1", "G2", "G3"]
n_before_outlier = len(df)
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR
    upper = Q3 + 3 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]
print(f"Outlier rows removed: {n_before_outlier - len(df)}")

# Validate value ranges for categorical ordinal columns
ordinal_ranges = {
    "studytime": (1, 4), "failures": (0, 3), "famrel": (1, 5),
    "freetime": (1, 5), "goout": (1, 5), "Dalc": (1, 5),
    "Walc": (1, 5), "health": (1, 5)
}
for col, (lo, hi) in ordinal_ranges.items():
    if col in df.columns:
        invalid = df[(df[col] < lo) | (df[col] > hi)]
        if len(invalid) > 0:
            print(f"Removing {len(invalid)} invalid rows in {col}")
            df = df[(df[col] >= lo) & (df[col] <= hi)]

print(f"\nClean dataset shape: {df.shape}")
print("\nMissing values after cleaning:", df.isnull().sum().sum(), "missing values")

# Save clean dataset
df.to_csv("student-mat-clean.csv", index=False, sep=";")
print("Clean dataset saved → student-mat-clean.csv")

# ==============================
# 3. CREATE TARGET VARIABLE
# ==============================

df["pass"] = np.where(df["G3"] >= 10, 1, 0)
df = df.drop(["G1", "G2", "G3"], axis=1)

X = df.drop("pass", axis=1)
y = df["pass"]

print("\nProcessed shape:", df.shape)
print("\nClass distribution (original):")
print(y.value_counts())

# ==============================
# 4. ENCODE CATEGORICAL FEATURES
# ==============================

X = pd.get_dummies(X, drop_first=True)

# ==============================
# 5. TRAIN-TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 6. SMOTE - CLASS IMBALANCE HANDLING
# ==============================

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("\nClass distribution after SMOTE:")
print(Counter(y_train_res))

# ==============================
# 7. INFORMATION GAIN TABLE
# ==============================

print("\n--- Information Gain (Entropy-based) for Top Features ---")

def entropy(labels):
    counts = np.bincount(labels)
    probs = counts / len(labels)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def information_gain(feature, labels):
    total_entropy = entropy(labels)
    unique_vals = np.unique(feature)
    weighted_entropy = 0
    for val in unique_vals:
        mask = feature == val
        subset_labels = labels[mask]
        weighted_entropy += (len(subset_labels) / len(labels)) * entropy(subset_labels)
    return total_entropy - weighted_entropy

ig_scores = {}
y_arr = y_train_res.values if hasattr(y_train_res, 'values') else np.array(y_train_res)
for col in X_train_res.columns:
    feat = X_train_res[col].values
    # Bin continuous features for IG calculation
    if feat.max() > 5:
        feat_binned = pd.cut(feat, bins=5, labels=False)
    else:
        feat_binned = feat
    ig_scores[col] = information_gain(feat_binned, y_arr)

ig_df = pd.DataFrame({
    "Feature": list(ig_scores.keys()),
    "Information Gain": [round(v, 6) for v in ig_scores.values()]
}).sort_values("Information Gain", ascending=False).reset_index(drop=True)
ig_df.index += 1

print("\nTop 15 Features by Information Gain:")
print(ig_df.head(15).to_string())

# ==============================
# 8. BASELINE DECISION TREE
# ==============================

tree_baseline = DecisionTreeClassifier(random_state=42)
tree_baseline.fit(X_train_res, y_train_res)

y_pred_baseline = tree_baseline.predict(X_test)

print("\n--- Baseline Decision Tree (No Depth Limit) ---")
print("Training Accuracy:", tree_baseline.score(X_train_res, y_train_res))
print("Test Accuracy:    ", accuracy_score(y_test, y_pred_baseline))
print("Precision:", precision_score(y_test, y_pred_baseline, zero_division=0))
print("Recall:   ", recall_score(y_test, y_pred_baseline, zero_division=0))
print("F1 Score: ", f1_score(y_test, y_pred_baseline, zero_division=0))
print("\nConfusion Matrix (Baseline):")
print(confusion_matrix(y_test, y_pred_baseline))
print("\nClassification Report (Baseline):")
print(classification_report(y_test, y_pred_baseline, target_names=["Fail", "Pass"]))

# ==============================
# 9. GRIDSEARCHCV - HYPERPARAMETER TUNING
# ==============================

param_grid = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=skf,
    scoring='f1_weighted',
    n_jobs=-1
)
grid_search.fit(X_train_res, y_train_res)

print("\n--- GridSearchCV Results ---")
print("Best Parameters:", grid_search.best_params_)
print("Best CV F1 (weighted):", round(grid_search.best_score_, 4))

best_model = grid_search.best_estimator_
best_depth = grid_search.best_params_['max_depth']

# ==============================
# 10. BEST MODEL EVALUATION
# ==============================

y_pred_best = best_model.predict(X_test)

print("\n--- Best Model (Tuned Decision Tree) ---")
print("Training Accuracy:", best_model.score(X_train_res, y_train_res))
print("Test Accuracy:    ", accuracy_score(y_test, y_pred_best))
print("Precision:", precision_score(y_test, y_pred_best, zero_division=0))
print("Recall:   ", recall_score(y_test, y_pred_best, zero_division=0))
print("F1 Score: ", f1_score(y_test, y_pred_best, zero_division=0))
print("\nConfusion Matrix (Best Model):")
print(confusion_matrix(y_test, y_pred_best))
print("\nClassification Report (Best Model):")
print(classification_report(y_test, y_pred_best, target_names=["Fail", "Pass"]))

# ==============================
# 11. GINI vs ENTROPY COMPARISON (depths 3, 5, 8)
# ==============================

print("\n--- Gini vs Entropy Comparison across Depths (3, 5, 8) ---")
depths_compare = [3, 5, 8]
comparison_results = []

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for i, depth in enumerate(depths_compare):
    tree_gini = DecisionTreeClassifier(max_depth=depth, criterion='gini', random_state=42)
    tree_entropy = DecisionTreeClassifier(max_depth=depth, criterion='entropy', random_state=42)

    tree_gini.fit(X_train_res, y_train_res)
    tree_entropy.fit(X_train_res, y_train_res)

    gini_pred = tree_gini.predict(X_test)
    entropy_pred = tree_entropy.predict(X_test)

    gini_acc  = accuracy_score(y_test, gini_pred)
    gini_f1   = f1_score(y_test, gini_pred, zero_division=0, average='weighted')
    gini_prec = precision_score(y_test, gini_pred, zero_division=0)
    gini_rec  = recall_score(y_test, gini_pred, zero_division=0)

    ent_acc  = accuracy_score(y_test, entropy_pred)
    ent_f1   = f1_score(y_test, entropy_pred, zero_division=0, average='weighted')
    ent_prec = precision_score(y_test, entropy_pred, zero_division=0)
    ent_rec  = recall_score(y_test, entropy_pred, zero_division=0)

    comparison_results.append({
        "Depth": depth, "Criterion": "Gini",
        "Train Acc": round(tree_gini.score(X_train_res, y_train_res), 4),
        "Test Acc": round(gini_acc, 4), "Precision": round(gini_prec, 4),
        "Recall": round(gini_rec, 4), "F1 (Weighted)": round(gini_f1, 4)
    })
    comparison_results.append({
        "Depth": depth, "Criterion": "Entropy",
        "Train Acc": round(tree_entropy.score(X_train_res, y_train_res), 4),
        "Test Acc": round(ent_acc, 4), "Precision": round(ent_prec, 4),
        "Recall": round(ent_rec, 4), "F1 (Weighted)": round(ent_f1, 4)
    })

    print(f"\ndepth={depth}:")
    print(f"  Gini    - Train: {tree_gini.score(X_train_res, y_train_res):.4f} | Test: {gini_acc:.4f} | Prec: {gini_prec:.4f} | Rec: {gini_rec:.4f} | F1: {gini_f1:.4f}")
    print(f"  Entropy - Train: {tree_entropy.score(X_train_res, y_train_res):.4f} | Test: {ent_acc:.4f} | Prec: {ent_prec:.4f} | Rec: {ent_rec:.4f} | F1: {ent_f1:.4f}")

    cm_gini    = confusion_matrix(y_test, gini_pred)
    cm_entropy = confusion_matrix(y_test, entropy_pred)

    print(f"\n  Confusion Matrix — Gini (depth={depth}):")
    print(cm_gini)
    print(f"  Confusion Matrix — Entropy (depth={depth}):")
    print(cm_entropy)

    ConfusionMatrixDisplay(cm_gini, display_labels=["Fail", "Pass"]).plot(ax=axes[0, i], colorbar=False)
    axes[0, i].set_title(f"Gini (depth={depth})\nAcc={gini_acc:.3f}  F1={gini_f1:.3f}", fontsize=11)

    ConfusionMatrixDisplay(cm_entropy, display_labels=["Fail", "Pass"]).plot(ax=axes[1, i], colorbar=False)
    axes[1, i].set_title(f"Entropy (depth={depth})\nAcc={ent_acc:.3f}  F1={ent_f1:.3f}", fontsize=11)

plt.suptitle("Confusion Matrices: Gini vs Entropy at Depths 3, 5, 8", fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('confusion_matrix_gini_entropy.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nConfusion matrix grid saved → confusion_matrix_gini_entropy.png")

comp_df = pd.DataFrame(comparison_results)
print("\n--- Gini vs Entropy Full Comparison Table ---")
print(comp_df.to_string(index=False))

# ==============================
# 12. STRATIFIED K-FOLD CROSS VALIDATION
# ==============================

cv_scores_acc = cross_val_score(best_model, X, y, cv=skf, scoring='accuracy')
cv_scores_f1  = cross_val_score(best_model, X, y, cv=skf, scoring='f1_weighted')

print("\n--- Stratified 5-Fold Cross-Validation (Best Model) ---")
print("CV Accuracy Scores:", np.round(cv_scores_acc, 4))
print("Mean CV Accuracy:   {:.4f} (+/- {:.4f})".format(cv_scores_acc.mean(), cv_scores_acc.std()))
print("CV F1 Scores:      ", np.round(cv_scores_f1, 4))
print("Mean CV F1:         {:.4f} (+/- {:.4f})".format(cv_scores_f1.mean(), cv_scores_f1.std()))

# ==============================
# 13. LEARNING CURVE
# ==============================

train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train_res, y_train_res,
    cv=skf, scoring='f1_weighted',
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

plt.figure(figsize=(9, 5))
plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training F1')
plt.plot(train_sizes, val_scores.mean(axis=1),   's-', label='Cross-Validation F1')
plt.fill_between(train_sizes,
    train_scores.mean(axis=1) - train_scores.std(axis=1),
    train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
plt.fill_between(train_sizes,
    val_scores.mean(axis=1) - val_scores.std(axis=1),
    val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
plt.xlabel('Training Set Size')
plt.ylabel('F1 Score (Weighted)')
plt.title('Learning Curve — Tuned Decision Tree')
plt.legend()
plt.tight_layout()
plt.savefig('learning_curve.png', dpi=150)
plt.close()
print("\nLearning curve saved → learning_curve.png")

# ==============================
# 14. TREE VISUALISATION
# ==============================

fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(
    best_model,
    feature_names=X.columns.tolist(),
    class_names=['Fail', 'Pass'],
    max_depth=3, filled=True, rounded=True, fontsize=9, ax=ax
)
plt.title('Decision Tree Visualisation (Top 3 Levels) — Tuned Model', fontsize=13)
plt.tight_layout()
plt.savefig('decision_tree_pruned.png', dpi=150, bbox_inches='tight')
plt.close()
print("Tree visualisation saved → decision_tree_pruned.png")

# ==============================
# 15. FEATURE IMPORTANCE PLOT
# ==============================

importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance_df.head(10).to_string(index=False))

plt.figure(figsize=(9, 5))
top10 = feature_importance_df.head(10)
plt.barh(top10["Feature"].values[::-1], top10["Importance"].values[::-1], color='steelblue')
plt.xlabel('Importance Score')
plt.title('Top 10 Feature Importances — Tuned Decision Tree')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.close()
print("Feature importance plot saved → feature_importance.png")

# ==============================
# 16. STUDENT RISK PROFILE TABLE
# ==============================

print("\n--- Student Risk Profile Tiers ---")
print("""
┌────────────┬──────────────────────────────────────────────────────┐
│ Risk Tier  │ Indicator Profile                                    │
├────────────┼──────────────────────────────────────────────────────┤
│ HIGH RISK  │ absences > 8  AND  failures >= 1                     │
│ MED RISK   │ absences 4-8  OR   goout >= 4  OR  studytime <= 1    │
│ LOW RISK   │ absences <= 3 AND  failures = 0 AND studytime >= 2   │
└────────────┴──────────────────────────────────────────────────────┘
""")

# ==============================
# 17. FINAL SUMMARY TABLE
# ==============================

print("\n--- Final Model Comparison Summary ---")
summary = pd.DataFrame({
    "Model": [
        "Baseline DT (no depth limit)",
        "Tuned DT - Gini   (depth={})".format(best_depth),
        "Tuned DT - Entropy (depth={})".format(best_depth),
        "Best Model (GridSearchCV)"
    ],
    "Train Acc": [
        round(tree_baseline.score(X_train_res, y_train_res), 4),
        round(comp_df[(comp_df['Criterion']=='Gini') & (comp_df['Depth']==best_depth)]['Train Acc'].values[0], 4),
        round(comp_df[(comp_df['Criterion']=='Entropy') & (comp_df['Depth']==best_depth)]['Train Acc'].values[0], 4),
        round(best_model.score(X_train_res, y_train_res), 4)
    ],
    "Test Acc": [
        round(accuracy_score(y_test, y_pred_baseline), 4),
        round(comp_df[(comp_df['Criterion']=='Gini') & (comp_df['Depth']==best_depth)]['Test Acc'].values[0], 4),
        round(comp_df[(comp_df['Criterion']=='Entropy') & (comp_df['Depth']==best_depth)]['Test Acc'].values[0], 4),
        round(accuracy_score(y_test, y_pred_best), 4)
    ],
    "F1 Weighted": [
        round(f1_score(y_test, y_pred_baseline, average='weighted', zero_division=0), 4),
        round(comp_df[(comp_df['Criterion']=='Gini') & (comp_df['Depth']==best_depth)]['F1 (Weighted)'].values[0], 4),
        round(comp_df[(comp_df['Criterion']=='Entropy') & (comp_df['Depth']==best_depth)]['F1 (Weighted)'].values[0], 4),
        round(f1_score(y_test, y_pred_best, average='weighted', zero_division=0), 4)
    ]
})
print(summary.to_string(index=False))
print("\nAll outputs generated successfully!")