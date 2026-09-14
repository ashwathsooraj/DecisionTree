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
# --- temporary test, will remove later ---
print(gini_impurity([0, 0, 0, 0]))      # expect 0.0
print(gini_impurity([0, 0, 1, 1]))      # expect 0.5
print(gini_impurity([0, 0, 1, 1, 1]))   # expect 0.48