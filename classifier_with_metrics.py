import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import csv
import os
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

POSITIVE_CLASS = 'COVID'
NEGATIVE_CLASS_1 = 'NORMAIS'
NEGATIVE_CLASS_2 = 'notCOVID'

# Slice inputs and outputs
def slice_data(dataset):
    # Slicing the input and output data
    input_data = dataset.iloc[:, :-1].values
    output_data = dataset.iloc[:, -1].values

    return [input_data, output_data]

# function to convert to binary output
def convert_to_binary_output(predicted):
    predicted_binary = []

    for predicted_class in predicted:
        if(predicted_class == POSITIVE_CLASS):
            predicted_binary.append(1)
        elif(predicted_class == NEGATIVE_CLASS_1):
            predicted_binary.append(0)
        elif(predicted_class == NEGATIVE_CLASS_2):
            predicted_binary.append(0)

    return predicted_binary

# Write result to formatted csv
def write_csv(sample_ids, predicted, probability_array, file_path):
    header = ['id_exame','class','prob']
    with open(file_path, 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL, dialect='excel')
        filewriter.writerow(header)
        for i in range(0, len(predicted)):
            result_row = [sample_ids[i],predicted[i],probability_array[i]]
            filewriter.writerow(result_row)

def count_per_class(output_data):
    original_count = np.unique(output_data, return_counts=True)
    classes = original_count[0]
    count = original_count[1]

    for i in range(0, len(original_count) + 1):
        print('Class {}, Count {}'.format(classes[i], count[i]))
    print('')


# Options
data = 'C:\\Users\\Fabio Barros\\Git\\covid-sp\\covid_train\\covid_train.csv'
classifier = 'svm'
resample = True
accuracy_array = []

# Load data
data_frame = pd.read_csv(data)

# Define the classifier
if(classifier == 'rf'):
    clf = RandomForestClassifier(criterion = "gini", min_samples_leaf = 10,  min_samples_split = 20, max_leaf_nodes = None, max_depth = 10)
elif(classifier == 'mlp'):
    clf = MLPClassifier(hidden_layer_sizes=(120), activation='logistic', verbose=False, early_stopping=True, validation_fraction=0.2)
elif(classifier == 'svm'):
    clf = SVC(gamma='auto', probability=True)


[input_data, output_data] = slice_data(data_frame)

kfold = KFold(n_splits=10, shuffle=True)
kfold_count = 1

for train_index, test_index in kfold.split(input_data, output_data):
    print('--------Started fold {} ----------'.format(kfold_count))

    # Slice inputs and outputs
    inputs_train, outputs_train = input_data[train_index], output_data[train_index]
    inputs_test, outputs_test = input_data[test_index], output_data[test_index]

    # Original class distribution
    count_per_class(outputs_train)

    # If resample flag is True, we need to resample the training dataset by generating new synthetic samples
    if resample:
        resampler = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=5, n_jobs=4)
        print("Resampling data")
        [input_data_train, output_data_train] = resampler.fit_resample(inputs_train, outputs_train)# Original class distribution
        print("Done resampling")
        # Resampled class distribution
        count_per_class(output_data_train)

    # Train the classifier
    print("Started training")
    clf = clf.fit(inputs_train, outputs_train)
    print("Finished training")

    # Predict
    print("Prediction")
    predicted = clf.predict(inputs_test)
    print("Finished prediction")


    print('--------Results for fold {} ----------'.format(kfold_count))
    #print(classification_report(outputs_test, predicted, labels=np.unique(outputs_test)))
    accuracy = accuracy_score(outputs_test, predicted)
    print('Accuracy Score: ' + str(accuracy))
    accuracy_array.append(accuracy)
    print('--------Finished fold {} ----------'.format(kfold_count))

    kfold_count+=1

print('--------Average result ----------'.format(kfold_count))
print('Avg Accuracy: {}'.format(np.mean(accuracy)))


