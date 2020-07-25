import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import csv
import os

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
        if(predicted_class == 'c1'):
            predicted_binary.append(1)
        elif(predicted_class == 'c2'):
            predicted_binary.append(0)
        elif(predicted_class == 'c3'):
            predicted_binary.append(0)

    return predicted_binary

# Write result to formatted csv
def write_csv(sample_ids, predicted, probability_array, file_path):
    header = ['id_exame','class','prob']
    with open(file_path + '\\result.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL, dialect='excel')
        filewriter.writerow(header)
        for i in range(0, len(predicted)):
            result_row = [sample_ids[i],predicted[i],probability_array[i]]
            filewriter.writerow(result_row)


# File path
train_data = 'Feature Matrix Train\\feature_matrix_train.csv'
test_data = 'Feature Matrix Test\\feature_matrix_test.csv'

# Load data
train_data_frame = pd.read_csv(train_data)
test_data_frame = pd.read_csv(test_data)

# Define the classifier
classifier = RandomForestClassifier(n_estimators=250, criterion='entropy',random_state=42)

# Slice inputs and outputs
[input_data_train, output_data_train] = slice_data(train_data_frame)
[input_data_test, sample_ids] = slice_data(test_data_frame)

# Train the classifier
classifier = classifier.fit(input_data_train, output_data_train)

# Predict
predicted = classifier.predict(input_data_test)
proba = classifier.predict_proba(input_data_test)

# Filter probability array
probability_c1 = proba[:,0]

# Convert to binary output
predicted = convert_to_binary_output(predicted)

result_dir = 'Result'
if not os.path.isdir(result_dir):
    os.mkdir(result_dir)

# Save result in csv
write_csv(sample_ids, predicted, probability_c1, result_dir)


