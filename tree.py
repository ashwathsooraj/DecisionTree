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
# --- temporary test, will remove later ---
X_test = np.array([[2.5], [1.0], [3.5], [0.5], [4.0], [1.5]])
y_test = np.array([1, 0, 1, 0, 1, 0])

feature, threshold, gini = best_split(X_test, y_test)
print(f"Best feature: {feature}, Best threshold: {threshold}, Best gini: {gini}")