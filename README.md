# DecisionTree 

## What's Implemented

- Gini impurity calculation
- Best-split search across features and thresholds
- Recursive tree building
- Prediction
- Training and evaluation on the Iris dataset
- Demonstration of overfitting using injected label noise
- Didn't understand what anything means after that 💔

## Project Structure

- tree.py — Core decision tree implementation (Gini impurity, splitting, recursive tree building, prediction)
- train_test.py — Loads the Iris dataset, trains the tree, evaluates accuracy, and demonstrates overfitting with noisy training labels
- WRITEUP.md — Write-up explaining the implementation and findings

## How to Run

1. Clone this repository:
```bash
git clone https://github.com/ashwathsooraj/DecisionTree.git
cd DecisionTree
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the training and overfitting demo:
```bash
python train_test.py
 ```

## Correctness Verification

Run the correctness harness to compare this implementation's predictions against scikit-learn's `DecisionTreeClassifier` (used here purely as a trusted reference for validation, not as part of the implementation itself):

```bash
python test_harness.py
```

This checks that, using matching settings (Gini criterion, same `max_depth` and `min_samples_split`), the custom tree's predictions agree with sklearn's reference implementation above a set threshold, and reports PASS or FAIL accordingly.

See `WRITEUP.md` for a full explanation of the implementation and results.