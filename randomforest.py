from decisiontree import DecisionTree
import numpy as np
from collections import Counter #working

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None, max_samples=None, random_state=None):
        """
        n_trees: Number of trees in the forest
        max_depth: Maximum depth of each decision tree
        min_samples_split: Minimum samples required to split a node
        n_features: Number of features to consider when looking for the best split
        max_samples: Maximum number of samples to draw from X to train each tree
        random_state: Seed for reproducibility
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.max_samples = max_samples
        self.trees = []
        
        # Optional random seed for reproducibility
        if random_state is not None:
            np.random.seed(random_state)

    # -------------------------
    # Train the Random Forest
    # -------------------------
    def fit(self, X, y):
        """Train the forest using bootstrap sampling and decision trees."""
        self.trees = []
        n_samples = X.shape[0]
        
        # Determine actual max_samples to use
        if self.max_samples is None:
            actual_max_samples = n_samples
        else:
            actual_max_samples = min(self.max_samples, n_samples)
        
        for _ in range(self.n_trees):
            # Bootstrap sample (random sampling with replacement)
            X_sample, y_sample = self._bootstrap_samples(X, y, actual_max_samples)
            
            # Create and train a new decision tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y, max_samples):
        """Return a bootstrap sample of the dataset with specified max_samples."""
        n_samples = X.shape[0]
        sample_size = min(max_samples, n_samples)
        idxs = np.random.choice(n_samples, sample_size, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        """Return the most common label in the list."""
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    # -------------------------
    # Prediction
    # -------------------------
    def predict(self, X):
        """Predict class labels for samples in X using majority vote."""
        # Get predictions from each tree → shape: (n_trees, n_samples)
        predictions = np.array([tree.predict(X) for tree in self.trees])

        # Swap axes → shape: (n_samples, n_trees)
        tree_preds = np.swapaxes(predictions, 0, 1)

        # Majority vote for each sample
        return np.array([self._most_common_label(pred) for pred in tree_preds])
    