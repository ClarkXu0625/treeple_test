import cupy as cp
import numpy as np
import os
# Set the NVRTC flags BEFORE importing cupy
#os.environ["CUPY_NVRTC_FLAGS"] = "--gpu-architecture=compute_120"  # Adjust to your GPU, or use "" to auto-detect


def gini(y):
    # Compute the Gini impurity for a set of labels y (assumed to be a cupy array)
    classes, counts = cp.unique(y, return_counts=True)
    prob = counts / counts.sum()
    return 1 - cp.sum(prob**2)

class GPUObliqueTree:
    def __init__(self, max_depth=5, min_samples_leaf=10, n_candidates=10):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_candidates = n_candidates
        self.tree = None

    def fit(self, X, y):
        # X and y should be cupy arrays.
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape

        # Stopping conditions: max depth, too few samples, or pure node
        if (depth >= self.max_depth) or (n_samples < self.min_samples_leaf) or (cp.unique(y).size == 1):
            majority_class = cp.mean(y).round().item()
            return {'leaf': True, 'class': majority_class}

        best_impurity = cp.inf
        best_split = None

        # Try a number of candidate hyperplanes
        for _ in range(self.n_candidates):
            # Generate a random hyperplane (weight vector)
            w = cp.random.randn(n_features)
            # Project data onto the hyperplane
            p = X.dot(w)
            # Choose threshold as the median of projections
            threshold = cp.median(p)

            # Create boolean masks for the split
            left_mask = p <= threshold
            right_mask = p > threshold

            if cp.sum(left_mask) == 0 or cp.sum(right_mask) == 0:
                continue

            impurity_left = gini(y[left_mask])
            impurity_right = gini(y[right_mask])
            impurity = (cp.sum(left_mask) * impurity_left + cp.sum(right_mask) * impurity_right) / n_samples

            if impurity < best_impurity:
                best_impurity = impurity
                best_split = {
                    'w': w,
                    'threshold': threshold,
                    'left_mask': left_mask,
                    'right_mask': right_mask
                }

        # If no valid split is found, return a leaf node.
        if best_split is None:
            majority_class = cp.mean(y).round().item()
            return {'leaf': True, 'class': majority_class}

        # Recursively build the left and right branches.
        left_tree = self._build_tree(X[best_split['left_mask']], y[best_split['left_mask']], depth + 1)
        right_tree = self._build_tree(X[best_split['right_mask']], y[best_split['right_mask']], depth + 1)

        return {
            'leaf': False,
            'w': best_split['w'],
            'threshold': best_split['threshold'],
            'left': left_tree,
            'right': right_tree
        }

    def _predict_one(self, x):
        # Traverse the tree for a single sample x (as a cupy array)
        node = self.tree
        while not node['leaf']:
            if cp.dot(x, node['w']) <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['class']

    def predict(self, X):
        # Ensure X is a cupy array.
        if not isinstance(X, cp.ndarray):
            X = cp.array(X)
        # Compute predictions for each sample.
        preds = cp.array([self._predict_one(x) for x in X])
        return cp.asnumpy(preds)

class GPUObliqueRandomForest:
    def __init__(self, n_estimators=10, max_depth=5, min_samples_leaf=10, n_candidates=10):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_candidates = n_candidates
        self.trees = []

    def fit(self, X, y):
        # Convert input data to cupy arrays if they are not already.
        if not isinstance(X, cp.ndarray):
            X = cp.array(X)
        if not isinstance(y, cp.ndarray):
            y = cp.array(y)
        n_samples = X.shape[0]
        self.trees = []

        for i in range(self.n_estimators):
            # Bootstrap sample the data.
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            tree = GPUObliqueTree(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, n_candidates=self.n_candidates)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Convert X to cupy array if necessary.
        if not isinstance(X, cp.ndarray):
            X_cp = cp.array(X)
        else:
            X_cp = X

        # Aggregate predictions from all trees.
        preds = []
        for tree in self.trees:
            preds.append(tree.predict(X_cp))
        preds = np.array(preds)
        # Majority vote across trees.
        maj_vote = np.round(np.mean(preds, axis=0)).astype(int)
        return maj_vote

# Example usage:
# if __name__ == '__main__':
#     # Create synthetic data for demonstration (binary classification)
#     np.random.seed(42)
#     n_samples = 1000
#     n_features = 20
#     X_cpu = np.random.randn(n_samples, n_features)
#     # Create labels with a simple linear rule (with noise)
#     y_cpu = (X_cpu[:, 0] + X_cpu[:, 1] > 0).astype(int)

#     # Initialize and train the GPU-accelerated oblique random forest
#     gpu_rf = GPUObliqueRandomForest(n_estimators=5, max_depth=5, min_samples_leaf=10, n_candidates=20)
#     gpu_rf.fit(X_cpu, y_cpu)

#     # Predict on the training data
#     predictions = gpu_rf.predict(X_cpu)
#     accuracy = np.mean(predictions == y_cpu)
#     print(f"Accuracy on synthetic data: {accuracy:.4f}")
