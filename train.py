from randomforest import RandomForest
import numpy as np         #maximum accuracy at (20,15,auto,true,190)
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load a dataset (Iris for example)
# iris = datasets.load_iris()
# X, y = iris.data, iris.target
df = pd.read_csv("heart.csv")

# Separate features (X) and target (y)
X = df.drop(columns=['target'])  # all columns except 'target'
y = df['target']
X =X.values
y = y.values
# 2. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Create and train the Random Forest model
clf = RandomForest(
    n_trees=15,           # Number of trees
    max_depth=10,         # Max depth of each tree
    min_samples_split=2,  # Min samples required to split
    n_features=4,      # Use all features
    random_state=42       # For reproducibility
)
clf.fit(X_train, y_train)

# 4. Make predictions 
y_pred = clf.predict(X_test)

# 5. Calculate accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}")

# 6. Show confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# 7. Show detailed classification report
report = classification_report(y_test, y_pred, target_names=['0','1'])
print("\nClassification Report:")
print(report)
