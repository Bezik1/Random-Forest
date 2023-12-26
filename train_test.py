from const.path import TRAINED_FOREST
from const.random_forest_config import TREE_NUMBER
from structure.RandomForest import RandomForest
from helpers.transform_output import transform_output
from data.data import training_data, test_data

X_train = training_data[:, :-1]
Y_train = training_data[:, -1].reshape(-1, 1)

test_data_inputs = test_data[:, :-1]

random_forest_classifier = RandomForest(TREE_NUMBER)
random_forest_classifier.fit(X_train, Y_train)
random_forest_classifier.save_forest(TRAINED_FOREST)
predictions = random_forest_classifier.predict(test_data_inputs)

[print(f'Target Class: {transform_output(data[1])} | Prediction Class: {transform_output(data[0][0])} for {data[0][1]*100 / random_forest_classifier.trees_number}%') for data in zip(predictions, test_data[:, 1])]