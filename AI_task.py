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
    f1_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ==============================
# 1. LOAD DATASET
# ==============================

df = pd.read_csv("student-mat.csv", sep=";")
print("Raw shape:", df.shape)

# ==============================
# 2. FIX DATA TYPES
# ==============================

df["G1"] = pd.to_numeric(df["G1"])
df["G2"] = pd.to_numeric(df["G2"])
df["G3"] = pd.to_numeric(df["G3"])

# ==============================
# 3. CREATE TARGET VARIABLE
# ==============================

df["pass"] = np.where(df["G3"] >= 10, 1, 0)
df = df.drop(["G1", "G2", "G3"], axis=1)

X = df.drop("pass", axis=1)
y = df["pass"]

print("Processed shape:", df.shape)
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
# 7. BASELINE DECISION TREE
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
# 8. GRIDSEARCHCV - HYPERPARAMETER TUNING
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

# ==============================
# 9. BEST MODEL EVALUATION
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
# 10. GINI vs ENTROPY COMPARISON
# ==============================

best_depth = grid_search.best_params_['max_depth']

tree_gini = DecisionTreeClassifier(
    max_depth=best_depth, criterion='gini', random_state=42
)
tree_entropy = DecisionTreeClassifier(
    max_depth=best_depth, criterion='entropy', random_state=42
)

tree_gini.fit(X_train_res, y_train_res)
tree_entropy.fit(X_train_res, y_train_res)

gini_pred    = tree_gini.predict(X_test)
entropy_pred = tree_entropy.predict(X_test)

print("\n--- Gini vs Entropy Comparison (max_depth={}) ---".format(best_depth))
print("Gini    - Test Accuracy: {:.4f} | F1: {:.4f}".format(
    accuracy_score(y_test, gini_pred),
    f1_score(y_test, gini_pred, zero_division=0)
))
print("Entropy - Test Accuracy: {:.4f} | F1: {:.4f}".format(
    accuracy_score(y_test, entropy_pred),
    f1_score(y_test, entropy_pred, zero_division=0)
))

# ==============================
# 11. STRATIFIED K-FOLD CROSS VALIDATION
# ==============================

cv_scores_acc = cross_val_score(best_model, X, y, cv=skf, scoring='accuracy')
cv_scores_f1  = cross_val_score(best_model, X, y, cv=skf, scoring='f1_weighted')

print("\n--- Stratified 5-Fold Cross-Validation (Best Model) ---")
print("CV Accuracy Scores:", np.round(cv_scores_acc, 4))
print("Mean CV Accuracy:   {:.4f} (+/- {:.4f})".format(
    cv_scores_acc.mean(), cv_scores_acc.std()))
print("CV F1 Scores:      ", np.round(cv_scores_f1, 4))
print("Mean CV F1:         {:.4f} (+/- {:.4f})".format(
    cv_scores_f1.mean(), cv_scores_f1.std()))

# ==============================
# 12. LEARNING CURVE
# ==============================

train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train_res, y_train_res,
    cv=skf,
    scoring='f1_weighted',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
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
# 13. TREE VISUALISATION (matplotlib, no system install needed)
# ==============================

fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(
    best_model,
    feature_names=X.columns.tolist(),
    class_names=['Fail', 'Pass'],
    max_depth=3,          # show top 3 levels only for readability
    filled=True,
    rounded=True,
    fontsize=9,
    ax=ax
)
plt.title('Decision Tree Visualisation (Top 3 Levels) — Tuned Model', fontsize=13)
plt.tight_layout()
plt.savefig('decision_tree_pruned.png', dpi=150, bbox_inches='tight')
plt.close()
print("Tree visualisation saved → decision_tree_pruned.png")

# ==============================
# 14. FEATURE IMPORTANCE PLOT
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
# 15. STUDENT RISK PROFILE TABLE
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
# 16. FINAL SUMMARY TABLE
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
        round(tree_gini.score(X_train_res, y_train_res), 4),
        round(tree_entropy.score(X_train_res, y_train_res), 4),
        round(best_model.score(X_train_res, y_train_res), 4)
    ],
    "Test Acc": [
        round(accuracy_score(y_test, y_pred_baseline), 4),
        round(accuracy_score(y_test, gini_pred), 4),
        round(accuracy_score(y_test, entropy_pred), 4),
        round(accuracy_score(y_test, y_pred_best), 4)
    ],
    "F1 Weighted": [
        round(f1_score(y_test, y_pred_baseline, average='weighted', zero_division=0), 4),
        round(f1_score(y_test, gini_pred,        average='weighted', zero_division=0), 4),
        round(f1_score(y_test, entropy_pred,     average='weighted', zero_division=0), 4),
        round(f1_score(y_test, y_pred_best,      average='weighted', zero_division=0), 4)
    ]
})
print(summary.to_string(index=False))