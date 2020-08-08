import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler, ADASYN
import os

train_data = 'C:/Users/Fabio Barros/Git/covid-sp/data/rydles_covid_train_59_fase2/rydles_covid_19_fase2_train.csv'
resample_algorithm = 'ros'
file_name = 'rydles_covid_19_fase2_train_'+resample_algorithm


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
train_data_frame = pd.read_csv(train_data)

# Split inputs and outputs
[input_data_train, output_data_train] = slice_data(train_data_frame)

# Resample data
train_data_frame = resample_data(input_data_train, output_data_train, define_resampler(resample_algorithm))

# Save resampled data in a CSV
directory = os.getcwd() + '/'+ file_name
if not os.path.isdir(directory):
    os.mkdir(directory)

train_data_frame.to_csv(directory + '/' + file_name + '.csv', index=False)

