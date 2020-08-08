import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler, ADASYN
import os
from sklearn.model_selection import train_test_split

data = 'C:/Users/Fabio Barros/Git/covid-sp/data/rydles_covid_train_59_fase2/rydles_covid_19_fase2_train.csv'
resample_algorithm = 'adasyn'
test_percentage = 0.2

train_file_name = 'rydles_covid_19_fase2_train_holdout_'+resample_algorithm
test_file_name = 'rydles_covid_19_fase2_test_holdout_'+str(test_percentage)

def slice_and_split_data_holdout(input_data, output_data, test_percentage):
    # Splitting the dataset in training/test using the Holdout technique
    inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(input_data, output_data,
                                                                              test_size=test_percentage,
                                                                              random_state=42)

    return [inputs_train, outputs_train, inputs_test, outputs_test]  # Return train and test data separately

def count_per_class(output_data):
    original_count = np.unique(output_data, return_counts=True)
    classes = original_count[0]
    count = original_count[1]

    for i in range(0, len(classes)):
        print('Class {}, Count {}'.format(classes[i], count[i]))
    print('')

def resample_data(input_data_train, output_data_train, resampler):

    # Original class distribution
    print('Original Class Distribution')
    count_per_class(output_data_train)

    # If resample flag is True, we need to resample the training dataset by generating new synthetic samples
    print("Resampling data")
    [input_data_train, output_data_train] = resampler.fit_resample(input_data_train,
                                                                   output_data_train)  # Original class distribution
    print("Done resampling")
    # Resampled class distribution
    count_per_class(output_data_train)

    train_data_frame = pd.DataFrame(input_data_train)
    train_data_frame['class'] = output_data_train

    return train_data_frame

# Slice inputs and outputs
def slice_data(dataset):
    # Slicing the input and output data
    input_data = dataset.iloc[:, :-1].values
    output_data = dataset.iloc[:, -1].values

    return [input_data, output_data]

def define_resampler(resampler_option):
    if(resampler_option == 'smote'):
        return SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42, n_jobs=4)
    elif(resampler_option == 'smote-enn'):
        return SMOTEENN(sampling_strategy='auto', random_state=42, n_jobs=4)
    elif(resampler_option == 'smote-tomek'):
        return SMOTETomek(sampling_strategy='auto', random_state=42, n_jobs=4)
    elif(resampler_option == 'borderline-smote'):
        return BorderlineSMOTE(sampling_strategy='auto', random_state=42, n_jobs=4)
    elif(resampler_option == 'adasyn'):
        return ADASYN(sampling_strategy='auto', random_state=42, n_jobs=4)
    elif(resampler_option == 'ros'):
        return RandomOverSampler(sampling_strategy='auto', random_state=42)


# Reading training data
data_frame = pd.read_csv(data)

# Split inputs and outputs
[input_data, output_data] = slice_data(data_frame)

# Apply holdout with a given percentage
[inputs_train, outputs_train, inputs_test, outputs_test] = slice_and_split_data_holdout(input_data, output_data, test_percentage)

# Resample data
train_data_frame_resampled = resample_data(inputs_train, outputs_train, define_resampler(resample_algorithm))

# Saving testing data without resampling
print('Test data distribution:')
count_per_class(outputs_test)
test_data_frame = pd.DataFrame(inputs_test)
test_data_frame['class'] = outputs_test

# Save training resampled data in a CSV
directory = os.getcwd() + '/'+ train_file_name
if not os.path.isdir(directory):
    os.mkdir(directory)

train_data_frame_resampled.to_csv(directory + '/' + train_file_name + '.csv', index=False)

# Save Testing data without resampling in a CSV
directory = os.getcwd() + '/'+ train_file_name
if not os.path.isdir(directory):
    os.mkdir(directory)

test_data_frame.to_csv(directory + '/' + test_file_name + '.csv', index=False)
