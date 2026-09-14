import numpy as np

def gini_impurity(y):
    """
    y: array of class labels for a group of samples, e.g. [0, 0, 1, 1, 1]
    Returns a number between 0 (pure) and up to ~0.5-ish for 2 classes (max mixed)
    """
    if len(y) == 0:
        return 0
    
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1 - np.sum(probabilities ** 2)
def split_dataset(X, y, feature_index, threshold):
    """
    X: 2D array of features (rows=samples, cols=features)
    y: 1D array of labels
    feature_index: which column to split on
    threshold: the value to split at
    
    Returns: left X, left y, right X, right y
    """
    left_mask = X[:, feature_index] <= threshold
    right_mask = ~left_mask   # everything not in left
    
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]
def best_split(X, y):
    """
    Try all features and thresholds, return the one with lowest weighted Gini.
    Returns: (best_feature_index, best_threshold, best_gini) or (None, None, None) if no split helps
    """
    n_samples, n_features = X.shape
    best_gini = float('inf')
    best_feature = None
    best_threshold = None
    
    for feature_index in range(n_features):
        thresholds = np.unique(X[:, feature_index])  # candidate split points
        
        for threshold in thresholds:
            left_X, left_y, right_X, right_y = split_dataset(X, y, feature_index, threshold)
            
            if len(left_y) == 0 or len(right_y) == 0:
                continue  # skip splits that don't actually divide the data
            
            # weighted average of the two groups' impurity
            weighted_gini = (len(left_y) / n_samples) * gini_impurity(left_y) + \
                             (len(right_y) / n_samples) * gini_impurity(right_y)
            
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature_index
                best_threshold = threshold
    
    return best_feature, best_threshold, best_gini
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # which feature this node splits on
        self.threshold = threshold          # the split value
        self.left = left                    # left child Node
        self.right = right                  # right child Node
        self.value = value                  # if this is a leaf, the predicted class
    
    def is_leaf(self):
        return self.value is not None   
def most_common_label(y):
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]


def build_tree(X, y, depth=0, max_depth=None, min_samples_split=2):
    n_samples = len(y)
    n_classes = len(np.unique(y))
    
    # Stopping conditions -> make a leaf
    if (n_classes == 1 or 
        n_samples < min_samples_split or 
        (max_depth is not None and depth >= max_depth)):
        leaf_value = most_common_label(y)
        return Node(value=leaf_value)
    
    feature_index, threshold, gini = best_split(X, y)
    
    if feature_index is None:  # no split improves things
        leaf_value = most_common_label(y)
        return Node(value=leaf_value)
    
    left_X, left_y, right_X, right_y = split_dataset(X, y, feature_index, threshold)
    
    left_child = build_tree(left_X, left_y, depth + 1, max_depth, min_samples_split)
    right_child = build_tree(right_X, right_y, depth + 1, max_depth, min_samples_split)
    
    return Node(feature_index=feature_index, threshold=threshold, left=left_child, right=right_child)
def predict_one(node, x):
    if node.is_leaf():
        return node.value
    
    if x[node.feature_index] <= node.threshold:
        return predict_one(node.left, x)
    else:
        return predict_one(node.right, x)


def predict(tree_root, X):
    return np.array([predict_one(tree_root, x) for x in X])
# --- temporary test, will remove later ---
X_test = np.array([[2.5], [1.0], [3.5], [0.5], [4.0], [1.5]])
y_test = np.array([1, 0, 1, 0, 1, 0])

tree = build_tree(X_test, y_test)
predictions = predict(tree, X_test)

print("True labels:     ", y_test)
print("Predicted labels:", predictions)