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
# --- Overfitting demonstration ---
np.random.seed(42)

def add_label_noise(y, noise_fraction=0.2):
    """Randomly flips a fraction of labels to simulate a noisy/harder dataset."""
    y_noisy = y.copy()
    n_samples = len(y)
    n_noisy = int(n_samples * noise_fraction)
    
    noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
    classes = np.unique(y)
    
    for idx in noisy_indices:
        # pick a random different class than the true one
        wrong_classes = classes[classes != y_noisy[idx]]
        y_noisy[idx] = np.random.choice(wrong_classes)
    
    return y_noisy

y_train_noisy = add_label_noise(y_train, noise_fraction=0.2)
# Train an UNRESTRICTED tree on the noisy training data
overfit_tree = build_tree(X_train, y_train_noisy, max_depth=None, min_samples_split=2)

train_preds_noisy = predict(overfit_tree, X_train)
test_preds_noisy = predict(overfit_tree, X_test)

# Compare against the TRUE labels (not noisy) to see real-world performance
train_acc_noisy = np.mean(train_preds_noisy == y_train_noisy)  # accuracy on the noisy labels it trained on
test_acc_noisy = np.mean(test_preds_noisy == y_test)            # accuracy on clean, unseen test data

print("\n--- Overfitting Demo (unrestricted tree, 20% noisy training labels) ---")
print(f"Train accuracy (on noisy labels): {train_acc_noisy:.4f}")
print(f"Test accuracy (on clean test set): {test_acc_noisy:.4f}")
print(f"Gap: {train_acc_noisy - test_acc_noisy:.4f}")
