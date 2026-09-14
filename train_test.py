import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tree import build_tree, predict

# Load the dataset
iris = load_iris()
X = iris.data      # shape: (150, 4) - 150 samples, 4 features
y = iris.target    # shape: (150,) - class labels 0, 1, 2

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))
# Train the tree
tree = build_tree(X_train, y_train, max_depth=None)

# Predict on both train and test sets
train_predictions = predict(tree, X_train)
test_predictions = predict(tree, X_test)

# Calculate accuracy
train_accuracy = np.mean(train_predictions == y_train)
test_accuracy = np.mean(test_predictions == y_test)

print(f"Train accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")