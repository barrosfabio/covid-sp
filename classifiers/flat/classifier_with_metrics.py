import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
import csv
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.ensemble import BaggingClassifier

POSITIVE_CLASS = 'COVID'
NEGATIVE_CLASS_1 = 'NORMAIS'
NEGATIVE_CLASS_2 = 'notCOVID'

# Options
data = 'C:\\Users\\Fabio Barros\\Git\\covid-sp\\covid_train_59\\covid_sp_train_59.csv'

classifier = 'rf'
resample = True
resample_algorithm = 'smote-enn'
accuracy_array = []

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

def define_classifier():
    if classifier == 'rf':
        return RandomForestClassifier(criterion="gini", n_estimators=150)
    elif classifier == 'mlp':
        return MLPClassifier(hidden_layer_sizes=(60), activation='logistic', verbose=False, early_stopping=True,
                             validation_fraction=0.2)
    elif classifier == 'svm':
        param_grid = {'C': [1, 10, 100, 1000], 'gamma': [1, 0.1, 0.001, 0.0001], 'kernel': ['linear', 'rbf']}
        grid = GridSearchCV(SVC(gamma='auto', probability=True), param_grid, refit=True)
        return grid
    elif classifier == 'calibrated-classifier':
        base_estimator = RandomForestClassifier(n_estimators=100)
        calibrated_forest = CalibratedClassifierCV(
        base_estimator = base_estimator)
        param_grid = {'base_estimator__max_depth': [2, 4, 6, 8]}
        search = GridSearchCV(calibrated_forest, param_grid, cv=5)
        return search
    elif classifier == 'bagging_rf':

        return BaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=100), n_estimators = 100, random_state = 42, warm_start=True)


def define_resampler():
    if resample_algorithm == 'smote':
        return SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=5, n_jobs=4)

    elif resample_algorithm == 'smote-enn':
        return SMOTEENN(sampling_strategy='auto', random_state=42, n_jobs=4)

    elif resample_algorithm == 'smote-tomek':
        return SMOTETomek(sampling_strategy='auto', random_state=42, n_jobs=4)

    elif resample_algorithm == 'random':
        return RandomOverSampler(sampling_strategy='auto', random_state=4)

def plot_confusion_matrix(cm, classes, image_name,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(15, 15))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)


    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(image_name)
    #plt.show(block=False)
    plt.close()



# Load data
data_frame = pd.read_csv(data)

# Define the classifier
clf = define_classifier()

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
        resampler = define_resampler()
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

    conf_matrix = confusion_matrix(outputs_test, predicted)
    conf_matrix_labels = np.unique(outputs_test)
    plot_confusion_matrix(conf_matrix, classes=conf_matrix_labels, image_name='Conf_Matrix_' + str(kfold_count),
                          normalize=True,
                          title='Confusion Matrix')

    kfold_count+=1

print('--------Average result ----------'.format(kfold_count))
print('Avg Accuracy: {}'.format(np.mean(accuracy_array)*100))


