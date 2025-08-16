from randomforest import RandomForest
from decisiontree import DecisionTree
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Create synthetic noisy dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=3,
    flip_y=0.05,
    random_state=42
)

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Random Forest
clf_rf = RandomForest(
    n_trees=20,
    max_depth=15,
    min_samples_split=5,
    n_features=None,
    random_state=42
)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)

acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"\nRandom Forest Accuracy: {acc_rf:.4f}")
print("Confusion Matrix (Random Forest):\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report (Random Forest):\n", classification_report(y_test, y_pred_rf))

# 4. Decision Tree
clf_dt = DecisionTree(
    max_depth=15,
    min_samples_split=5,
    n_features=None
)
clf_dt.fit(X_train, y_train)
y_pred_dt = clf_dt.predict(X_test)

acc_dt = accuracy_score(y_test, y_pred_dt)
print(f"\nDecision Tree Accuracy: {acc_dt:.4f}")
print("Confusion Matrix (Decision Tree):\n", confusion_matrix(y_test, y_pred_dt))
print("Classification Report (Decision Tree):\n", classification_report(y_test, y_pred_dt))
