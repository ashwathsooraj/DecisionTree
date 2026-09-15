# Write-up: Decision Tree from Scratch

## 1. Overview

This project implements a decision tree classifier from scratch using only NumPy for numerical operations. No part of the tree-building or splitting logic uses scikit-learn's `DecisionTreeClassifier` or any other pre-built tree implementation.

The implementation covers:
1. Building a decision tree using Gini impurity as the splitting criterion
2. Supporting classification (with an explanation for why regression was left out)
3. Demonstrating overfitting by training an unrestricted tree on noisy data

## 2. Gini Impurity

To decide how to split data at each node, the tree needs a way to measure how "mixed" a group of labels is. This implementation uses Gini impurity, defined as:

    Gini = 1 - Σ(p_i)²

where `p_i` is the proportion of samples belonging to class `i` in a given group.

- A group with only one class present has Gini = 0 (perfectly pure)
- A group evenly split between classes has a higher Gini value (maximally mixed)

For example, a group of 5 samples with 2 of class 0 and 3 of class 1:
- p_0 = 2/5 = 0.4, p_1 = 3/5 = 0.6
- Gini = 1 - (0.4² + 0.6²) = 1 - (0.16 + 0.36) = 0.48

https://www.youtube.com/watch?v=u4IxOk2ijSs - I watched this video to give me a brief introduction into Gini impurity.

This is implemented in `gini_impurity()` in `tree.py`, using `numpy.unique` to count class occurrences and computing the formula directly.

## 3. Finding the Best Split

At the most basic level, a Decision Tree is just a bunch of Yes/No questions. Since Yes/No questions are so "narrow" in its usage, we need to ask the "best possible question" so that we can extract the most information out of this Yes/No. In essence, this "best possible question" is called the Best Split.

At each node, the tree needs to decide which feature to split on and at what threshold. This is done by brute-force search: for every feature, and every unique value of that feature present in the data, the algorithm:

1. Splits the data into a "left" group (feature value ≤ threshold) and a "right" group
   (feature value > threshold)
2. Computes the Gini impurity of each resulting group
3. Computes a weighted average of the two impurities, weighted by group size:

    weighted_gini = (n_left / n_total) * Gini(left) + (n_right / n_total) * Gini(right)

4. Keeps track of whichever (feature, threshold) pair produces the lowest weighted Gini

The split with the lowest weighted Gini is chosen, because it produces the two "purest" possible child groups, meaning the split does the best job of separating the classes.

The Best Split requires us to split the entire dataset into 2 parts such that the 2 parts are the most different from each other (Thus, we get the most information out of the Yes/No). It must be split in such a way that one set has one species and another set has another species (Ideal case, which would require multiple Yes/No questions in reality). 
Having a mixed dataset after the split means that we have learnt nothing new from the Yes/No question that we asked. We essentially just have the same dataset before the Yes/No just on a smaller scale.
This "purity" of each set can be established using the Gini impurity. If the Gini index is high, it means that the dataset in each group is highly mixed (undesirable) and a lower Gini index means a purer split (desirable).

For example, let us consider there are 10 flowers infront of us which have varying petal length and width (These are called features). The Best Split here would be: What is the best Yes/No question about these flowers that can best separate them by their species.
Some questions we can ask to split them are: Is the petal length >= 2 cm? Is the petal width <= 1cm? (These are called thresholds, and these criteria are the gist of the Iris training data which will be used later on to train this model).
Now, we can iterate it multiple times by putting the petal length criteria to be 1 cm, 1.5 cm, 2 cm, 2.5 cm (similarly for petal width too), and then use the weighted mean formula as shown above to find which case has the lowest Gini index.
By putting multiple values of length and width, we are brute-forcing until we get the lowest index by weighted mean of both groups. Thus, it is called a brute-force search.

This is implemented in `best_split()` in `tree.py`.

## 4. Building the Tree Recursively

The tree is built recursively via `build_tree()`. At each call:

1. Check stopping conditions: If the current group is already pure (only one class present), too small to split further (below `min_samples_split`), or has reached `max_depth`, the function creates a **leaf node** that predicts the most common class in that group (`most_common_label()`)
2. Otherwise, find the best split using `best_split()`
3. Split the data into left and right groups
4. Recursively call `build_tree()` on each group to build the left and right subtrees
5. Return a **decision node** storing the chosen feature, threshold, and pointers to the left/right subtrees

This recursive structure means the tree grows downward, each level splitting the data further, until every branch terminates in a leaf.

Prediction (`predict_one()`) mirrors this recursively: starting at the root, check the relevant feature against the node's threshold, move to the left or right child accordingly, and repeat until a leaf is reached, the leaf's stored value is the prediction.

In simple words, its just a step added to ensure that each branch terminates. So after one split, if we notice that the group still isn't pure, we move onto another split and so on recursively until it becomes pure. While this isn't the only reason that the branch terminates, the group becoming pure is the "hopeful" outcome. The other 2 reasons are failsafes of sort: one is when the group has too few samples worth splitting further (`min_samples_split`), and the other is when the branch has reached a maximum possible depth (`max_dept`).

## 5. Classification vs. Regression (4.2)

This implementation supports **classification only**. Regression was not implemented.

**Reason:** Regression trees require a different splitting criterion, instead of Gini impurity (which measures class mixing) or entropy.
The difference between Classification and Regression is that: Classification is discrete, as in it splits into one out of 2/3/n categories, while Regression is continuous in nature, as in the output received for "What price is this house" can be any real number (not any particular category or class).
Due to the nature of classification, indexes like entropy or Gini impurity can be used as differentiating factors to put stuff into different categories but we cannot use that same logic for regression. 
Some reasons why I used Classification over Regression is that: 
- Gini impurity and entropy being mentioned in the task document, which are generally used to split categorically, made me feel that I would only be able to implement these properly into this project if I used Classification.
- Decision Trees work on the principle of splitting things into categories or "buckets" repeatedly until it is sorted. It is not a smooth and continuous process which requires exact values, which Regression does.
- In hindsight, I should have spent time to properly implement a Regression model in some or the other way, but I frankly just did not have enough time. It would have required an entirely different set of functions from Classification rather than just a slight tweak on that model which would have been insanely time-consuming. I wanted to give a proper interpretation of Classification rather than a half-assed effort on Regression.

## 6. Testing on the Iris Dataset

The tree was trained and evaluated on the Iris dataset (150 samples, 4 numeric features, 3 classes), split into 70% training / 30% test data using `train_test_split` with a fixed random seed for reproducibility.

An unrestricted tree (no depth limit) trained on clean Iris data achieved:
- Training accuracy: 100%
- Test accuracy: 95.56%

This confirms the tree correctly learns and generalizes on a real classification
task.

## 7. Demonstrating Overfitting (4.3)

Overfitting is when a model basically "byhearts" the training data that was fed to it. It familiarises itself with all the quirks and randomness of the training data alone rather than understanding the pattern. This becomes an issue when it is fed with test data. Obviously, the test data will have its own set of quirks and randomness which is different from the training data, and the model will likely fail to adapt to it.

It's kind of like byhearting questions for a tute quiz, and then one of the questions comes ditto same as the byhearted ones but with different values/conditions and you fail to do it, then you claim that the question was out of syllabus. 

Iris is a relatively clean dataset, so an unrestricted tree doesn't overfit dramatically on it by default. To clearly demonstrate overfitting, artificial noise was introduced: 20% of the training labels were randomly flipped to an incorrect class using `add_label_noise()` in `train_test.py`, while the **test set remained clean/unmodified**.

An unrestricted tree (`max_depth=None`, `min_samples_split=2`) was then trained on this noisy training data. Results:

- Training accuracy (on noisy labels): 100%
- Test accuracy (on clean, unseen test data): 71.11%
- Gap: ~28.9 percentage points

**Interpretation:** Because the tree had no depth restriction, it kept splitting until every leaf was pure, including leaves that only existed to isolate the artificially mislabeled samples. This means the tree  memorized the specific noise/errors present in the training set rather than the underlying pattern. Because that noise was random and not present in the test set, this memorization didn't help and it produced a large accuracy gap between training and test performance.

This is the main sign of overfitting: near-perfect performance on data the model has already seen, but a significant drop in performance on new data.

## 8. Summary

This implementation demonstrates the full core mechanics of a decision tree classifier, impurity-based splitting, recursive tree construction, and prediction, built entirely from scratch with NumPy. It was validated against a real dataset (Iris), and used to concretely demonstrate overfitting by contrasting the behavior of an unrestricted tree on clean versus artificially noised training data. Classification was implemented; regression was intentionally left out due to time constraints, with the reasoning explained above.

## 9. NOTE

My biggest issue with this was GitHub. It was a mess trying to understand when to push and when to commit, how commits work, whether commit saves the progress of only one branch or all the branches, how to delete a commit etc. I had so many random and unnecessary commits that I ended up getting irritated and deleting my previous reposistory, made a new one and copy pasted the code from the other project to this. I know this beats the purpose of having commits because you may not be able to see which errors I ran into, so I apologise for that.

I could not get past 4.3, because of time constraints (once again) and because I had a tough time wrapping my mind around concepts past this. Maybe my current interpretations are shallow, but I will continue to explore this domain and try to get past 4.3.

Honestly speaking, I am an amateur at coding and do not have the expertise to implement all this on my own. Because of that, I had to resort to using AI to guide me through explanations and codes (I don't want to pretend as if I did not use AI for implementation). I presume it was fine as the task document mentioned that we can use AI for help.