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
# --- temporary test, will remove later ---
X_test = np.array([[1, 5], [2, 3], [3, 8], [4, 1]])
y_test = np.array([0, 0, 1, 1])

left_X, left_y, right_X, right_y = split_dataset(X_test, y_test, feature_index=0, threshold=2)

print("Left X:", left_X)
print("Left y:", left_y)
print("Right X:", right_X)
print("Right y:", right_y)