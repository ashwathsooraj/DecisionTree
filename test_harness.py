import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from tree import build_tree, predict

# Load data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

max_depth = 3
min_samples_split = 2

# Your implementation
my_tree = build_tree(X_train, y_train, max_depth=max_depth, min_samples_split=min_samples_split)
my_preds = predict(my_tree, X_test)
my_accuracy = np.mean(my_preds == y_test)

# Trusted reference: sklearn's DecisionTreeClassifier, same settings
sklearn_tree = DecisionTreeClassifier(criterion='gini', max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
sklearn_tree.fit(X_train, y_train)
sklearn_preds = sklearn_tree.predict(X_test)
sklearn_accuracy = np.mean(sklearn_preds == y_test)

agreement = np.mean(my_preds == sklearn_preds)

print("=== Correctness Harness: Custom Tree vs. sklearn DecisionTreeClassifier ===")
print(f"My tree accuracy:      {my_accuracy:.4f}")
print(f"sklearn tree accuracy: {sklearn_accuracy:.4f}")
print(f"Prediction agreement:  {agreement:.4f}")

# Pass/fail check: require close agreement with the trusted reference
TOLERANCE = 0.90  # at least 90% of predictions should match sklearn's
if agreement >= TOLERANCE:
    print(f"\nPASS: Predictions agree with sklearn reference ({agreement:.2%} >= {TOLERANCE:.0%})")
else:
    print(f"\nFAIL: Predictions disagree too much with sklearn reference ({agreement:.2%} < {TOLERANCE:.0%})")