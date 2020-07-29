import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import csv
import os
from imblearn.over_sampling import SMOTE
import numpy as np

POSITIVE_CLASS = 'COVID'
NEGATIVE_CLASS_1 = 'NORMAIS'
NEGATIVE_CLASS_2 = 'notCOVID'

# Options
CSV_SPACER = ";"

train_data_path = 'T:/0 - PUCPR/Mestrado/0 - COVID/TestandoWeka/covid_train.csv'
test_data_path = 'T:/0 - PUCPR/Mestrado/0 - COVID/TestandoWeka/covid_test.csv'
classifier = "rf" #rf, mlp or svm
resample = False


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
        if predicted_class == POSITIVE_CLASS:
            predicted_binary.append(1)
        elif predicted_class == NEGATIVE_CLASS_1:
            predicted_binary.append(0)
        elif predicted_class == NEGATIVE_CLASS_2:
            predicted_binary.append(0)

    return predicted_binary


# Write result to formatted csv
def write_csv(sample_ids, predicted, probability_array, file_path):
    header = ['id_exame', 'class', 'prob']

    with open(file_path, 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=CSV_SPACER, quotechar='|', quoting=csv.QUOTE_MINIMAL,
                                dialect='excel')
        filewriter.writerow(header)

        for i in range(0, len(predicted)):
            # print("CLASSE PREVISTA -- " + str(predicted[i]))
            if predicted[i] == 1:
                result_row = [sample_ids[i], predicted[i], probability_array[i]]
            else:
                result_row = [sample_ids[i], predicted[i]]

            filewriter.writerow(result_row)


def count_per_class(output_data):
    original_count = np.unique(output_data, return_counts=True)
    classes = original_count[0]
    count = original_count[1]

    for i in range(0, len(original_count) + 1):
        print('Class {}, Count {}'.format(classes[i], count[i]))
    print('')


def define_classifier():
    if classifier == 'rf':
        return RandomForestClassifier(criterion="gini", min_samples_leaf=10, min_samples_split=20, max_leaf_nodes=None,
                                      max_depth=10)
    elif classifier == 'mlp':
        return MLPClassifier(hidden_layer_sizes=(60), activation='logistic', verbose=False, early_stopping=True,
                             validation_fraction=0.2)
    elif classifier == 'svm':
        return SVC(gamma='auto', probability=True)


# Load data
train_data_frame = pd.read_csv(train_data_path)
test_data_frame = pd.read_csv(test_data_path)

# Define the classifier
clf = define_classifier()

# Slice inputs and outputs
[input_data_train, output_data_train] = slice_data(train_data_frame)
[input_data_test, sample_ids] = slice_data(test_data_frame)

# Original class distribution
count_per_class(output_data_train)

# If resample flag is True, we need to resample the training dataset by generating new synthetic samples
if resample:
    resampler = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=5, n_jobs=4)
    print("Resampling data")
    [input_data_train, output_data_train] = resampler.fit_resample(input_data_train, output_data_train)  # Original class distribution
    print("Done resampling")
    # Resampled class distribution
    count_per_class(output_data_train)

# Train the classifier
print("Started training")
clf = clf.fit(input_data_train, output_data_train)
print("Finished training")

# Predict
print("Prediction")
predicted = clf.predict(input_data_test)
proba = clf.predict_proba(input_data_test)
print("Finished prediction")

# Filter probability array
probability_c1 = proba[:, 0]

# Convert to binary output
predicted = convert_to_binary_output(predicted)

result_dir = 'Result'
if not os.path.isdir(result_dir):
    os.mkdir(result_dir)

file_path = result_dir + '\\result_' + classifier + '_resample_' + str(resample) + '.csv'

# Save result in csv
write_csv(sample_ids, predicted, probability_c1, file_path)

print('Done writing result to CSV.')
