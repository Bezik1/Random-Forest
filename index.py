import numpy as np

from const.path import TRAINED_FOREST
from const.random_forest_config import TREE_NUMBER
from structure.RandomForest import RandomForest
from helpers.transform_output import transform_output
from data.data import training_data, test_data

X_train = training_data[:, :-1]
Y_train = training_data[:, -1].reshape(-1, 1)

test_data_inputs = test_data[:, :-1]

random_forest_classifier = RandomForest(TREE_NUMBER)
random_forest_classifier.load_forest(TRAINED_FOREST)

while True:
    x = int(input("Light wave length(nm):"))
    prediction = random_forest_classifier.predict(np.array([[x]]))
    print(f'{transform_output(prediction[0][0])} for {prediction[0][1]*100 / random_forest_classifier.trees_number}%')