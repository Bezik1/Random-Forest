import numpy as np
import random

from structure.DecisionTree import DecisionTree

class RandomForest:
    def __init__(self, trees_number) -> None:
        self.trees_number = trees_number

        self.trees = [DecisionTree(min_samples_split=3, max_depth=100)
                      for t in range(self.trees_number)]
    
    def fit(self, X, Y):
        num_samples = X.shape[0]
        for tree in self.trees:
            indices = [random.randint(0, num_samples - 1) for _ in range(num_samples)]
            X_subset, Y_subset = X[indices], Y[indices]
            tree.fit(X_subset, Y_subset)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees]).T

        return [self.most_common(tree_predictions) for tree_predictions in predictions]
    
    def save_forest(self, filename):
        [tree.save_tree(f'{filename}-{str(i)}.json') for i, tree in enumerate(self.trees)]
    
    def load_forest(self, filename):
        [tree.load_tree(f'{filename}-{str(i)}.json') for i, tree in enumerate(self.trees)]

    def most_common(self, array):
        flattened_array = array.flatten()

        unique_values, frequencies = np.unique(flattened_array, return_counts=True)
        most_common_value = unique_values[np.argmax(frequencies)]
        occurrences = np.count_nonzero(flattened_array == most_common_value)

        return most_common_value, occurrences