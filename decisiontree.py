import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        """
        feature: index of feature used for split
        threshold: value of the feature to split on
        left: left child Node
        right: right child Node
        value: label (only for leaf nodes)
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        """Check if this node is a leaf."""
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        """
        min_samples_split: Minimum number of samples to allow further splitting
        max_depth: Maximum depth of the tree
        n_features: Number of features to consider when looking for the best split
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        """
        X: Feature matrix (numpy array)
        y: Target vector (numpy array)
        """
        # FIXED: Ensure n_features doesn't exceed available features
        max_available = X.shape[1]
        if self.n_features is None:
            self.n_features = max_available
        else:
            self.n_features = min(max_available, max(1, self.n_features))
        
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        """Recursive function to grow the tree."""
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # FIXED: Ensure we don't try to select more features than available
        feat_to_select = min(self.n_features, n_feats)
        if feat_to_select <= 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
            
        # Select random subset of features
        feat_idxs = np.random.choice(n_feats, feat_to_select, replace=False)

        # Find the best feature and threshold to split
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # If no split gives positive information gain → make leaf
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Perform the split
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        
        # FIXED: Ensure we have samples for both children
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
            
        # Recursively grow left and right children
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        """Return the best feature index and threshold for splitting."""
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        """Calculate information gain from a potential split."""
        parent_entropy = self._entropy(y)

        # Split the dataset into left and right parts
        left_idxs, right_idxs = self._split(X_column, threshold)

        # If no data in either side → no gain
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Weighted average of child entropies
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # Information Gain = Entropy before split - Entropy after split
        return parent_entropy - child_entropy

    def _split(self, X_column, split_thresh):
        """Split data into left/right indices based on threshold."""
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        """Calculate entropy of label array."""
        if len(y) == 0:
            return 0
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        """Return the most frequent class label in y."""
        if len(y) == 0:
            return 0
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        """Predict labels for all samples in X."""
        if self.root is None:
            return np.zeros(X.shape[0], dtype=int)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """Recursive function to traverse tree for a single sample."""
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)