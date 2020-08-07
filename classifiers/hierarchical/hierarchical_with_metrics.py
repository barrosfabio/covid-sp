from node import Node
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
import numpy as np
import csv
import os
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTEENN, SMOTETomek

POSITIVE_CLASS = 'COVID'
NEGATIVE_CLASS_1 = 'NORMAIS'
NEGATIVE_CLASS_2 = 'notCOVID'
INTERMEDIATE_NEGATIVE_CLASS = 'NOT_NORMAL'

# Options
CSV_SPACER = ";"

data = 'C:\\Users\\Fabio Barros\\Git\\covid-sp\\covid_train_59\\covid_sp_train_59.csv'
classifier = "rf" #rf, mlp or svm
resample = True
local_resample = False
resample_algorithm = 'smote-tomek'
result_dir = '../../Result_Hierarchical'

class Node:
    class_name = None
    is_leaf = False
    data = pd.DataFrame()
    left = None
    right = None
    local_clf = None

    def __init__(self, class_name):
        self.class_name = class_name

    def set_data(self, data):
        self.data = data

    def set_new_child(self, child):
        self.children.append(Node(child))

    def is_parent(self, is_parent):
        self.is_parent = is_parent

class Result:

    def __init__(self, predicted_class, proba):
        self.predicted_class = predicted_class
        self.proba = proba

    def set_proba(self, proba):
        self.proba = proba

# Slice inputs and outputs
def slice_data(dataset):
    # Slicing the input and output data
    input_data = dataset.iloc[:, :-1].values
    output_data = dataset.iloc[:, -1].values

    return [input_data, output_data]

def define_classifier():
    if classifier == 'rf':
        return RandomForestClassifier(criterion="gini", n_estimators=150)
    elif classifier == 'mlp':
        return MLPClassifier(hidden_layer_sizes=(60), activation='logistic', verbose=False, early_stopping=True,
                             validation_fraction=0.2)
    elif classifier == 'svm':
        return SVC(gamma='auto', probability=True)

def create_class_tree():

    tree = Node('R')
    normal = Node(NEGATIVE_CLASS_1)
    not_normal = Node(INTERMEDIATE_NEGATIVE_CLASS)
    altered_not_covid = Node(NEGATIVE_CLASS_2)
    covid = Node(POSITIVE_CLASS)
    normal.is_leaf = True
    altered_not_covid.is_leaf = True
    covid.is_leaf = True
    not_normal.left = altered_not_covid
    not_normal.right = covid
    tree.right = not_normal
    tree.left = normal

    return tree

def relabel_to_current_class(class_name, relabeled_data_frame):
    relabeled_data_frame['class'] = class_name
    return relabeled_data_frame

def retrieve_data_lcpn(tree, data_frame):
    class_data = pd.DataFrame()
    if tree.is_leaf == True:
        return data_frame[data_frame['class']==tree.class_name]
    else:
        class_data = class_data.append(retrieve_data_lcpn(tree.left, data_frame))
        class_data = class_data.append(retrieve_data_lcpn(tree.right, data_frame))

        if local_resample == True:
            print('------------Local Distribution for class: {}----------------'.format(tree.class_name))
            [input_data, output_data] = slice_data(class_data)
            class_data = resample_data(input_data, output_data)
            print('------------------------------------------------------------')

        tree.data = class_data

        # Rename to the current class before returning to parent class
        class_data_relabeled = relabel_to_current_class(tree.class_name, class_data.copy())

        return class_data_relabeled

def train_lcpn(tree):
    if tree.is_leaf == True:
        return
    else:
        # Will train only for parent node classes
        [input_data_train, output_data_train] = slice_data(tree.data)

        clf = clone(define_classifier())
        trained_clf = clf.fit(input_data_train, output_data_train)
        tree.local_clf = trained_clf

        train_lcpn(tree.left)
        train_lcpn(tree.right)

        return

def prediction_proba(data, model):

    data = data.reshape(1, -1)
    predicted = model.predict(data)
    proba = model.predict_proba(data)
    proba = proba[0]

    possible_classes = model.classes_
    # Find index of predicted class and save this index only
    index = np.where(possible_classes == predicted)
    index = index[0]

    proba_predicted_class = proba[index[0]]
    result = Result(predicted, proba_predicted_class)
    return result


def predict_lcpn(row, tree):

    if tree.left is not None or tree.right is not None:

        result = prediction_proba(row, tree.local_clf)

        if result.predicted_class[0] == tree.left.class_name:
            prediction_result = predict_lcpn(row, tree.left)
        elif result.predicted_class[0] == tree.right.class_name:
            prediction_result = predict_lcpn(row, tree.right)

        # If proba is None, that means the last prediction was a leaf node, as we want
        if(prediction_result.proba is None):
            prediction_result.set_proba(result.proba)

        return prediction_result

    else:
        result = Result(tree.class_name, None)
        return result

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
    header = ['id_exame', 'class', 'prob']

    with open(file_path, 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=CSV_SPACER, quotechar='|', quoting=csv.QUOTE_MINIMAL,
                                dialect='excel')
        filewriter.writerow(header)

        for i in range(0, len(predicted)):

            if predicted[i] == 1:
                result_row = [sample_ids[i], predicted[i], probability_array[i]]
            else:
                result_row = [sample_ids[i], predicted[i]]

            filewriter.writerow(result_row)

def count_per_class(output_data):
    original_count = np.unique(output_data, return_counts=True)
    classes = original_count[0]
    count = original_count[1]

    for i in range(0, len(classes)):
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
    plt.figure(figsize=(20, 20))
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

def resample_data(input_data_train, output_data_train):

    # Original class distribution
    count_per_class(output_data_train)

    # If resample flag is True, we need to resample the training dataset by generating new synthetic samples
    resampler = define_resampler()
    print("Resampling data")
    [input_data_train, output_data_train] = resampler.fit_resample(input_data_train,
                                                                   output_data_train)  # Original class distribution
    print("Done resampling")
    # Resampled class distribution
    count_per_class(output_data_train)

    train_data_frame = pd.DataFrame(input_data_train)
    train_data_frame['class'] = output_data_train

    return train_data_frame

def define_resampler():
    if resample_algorithm == 'smote':
        return SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=5, n_jobs=4)

    elif resample_algorithm == 'smote-enn':
        return SMOTEENN(sampling_strategy='auto', random_state=42, n_jobs=4)

    elif resample_algorithm == 'smote-tomek':
        return SMOTETomek(sampling_strategy='auto', random_state=42, n_jobs=4)

    elif resample_algorithm == 'random':
        return RandomOverSampler(sampling_strategy='auto', random_state=4)

# Load data
data_frame = pd.read_csv(data)
[input_data, output_data] = slice_data(data_frame)
accuracy_array = []

kfold = KFold(n_splits=5, shuffle=True)
kfold_count = 1

for train_index, test_index in kfold.split(input_data, output_data):
    print('----------Started fold {} ----------'.format(kfold_count))
    # Slice inputs and outputs
    input_data_train, output_data_train = input_data[train_index], output_data[train_index]

    # Original class distribution
    count_per_class(output_data_train)

    # If resample flag is True, we need to resample the training dataset by generating new synthetic samples
    if resample_data == True:
        train_data_frame = resample_data(input_data_train, output_data_train)
    else:
        train_data_frame = pd.DataFrame(input_data_train)
        train_data_frame['class'] = output_data_train

    test_data_frame, outputs_data_test = input_data[test_index], output_data[test_index]

    # Load Classifier
    clf = define_classifier()

    class_tree = create_class_tree()
    retrieve_data_lcpn(class_tree, train_data_frame)

    # Train
    train_lcpn(class_tree)

    # Predict
    prediction = []
    proba_array = []

    # Predict the class for each row
    for input_test_row in test_data_frame:
        prediction_result = predict_lcpn(input_test_row, class_tree)
        prediction.append(prediction_result.predicted_class)
        proba_array.append(prediction_result.proba)

    # Calculating Metrics
    print('--------Results for fold {} ----------'.format(kfold_count))
    accuracy = accuracy_score(outputs_data_test, prediction)
    accuracy_array.append(accuracy)
    print('Accuracy Score: ' + str(accuracy))
    print('Finished Fold {}'.format(kfold_count))
    print('--------------------------------------\n')

    conf_matrix = confusion_matrix(outputs_data_test, prediction)
    conf_matrix_labels = np.unique(outputs_data_test)
    plot_confusion_matrix(conf_matrix, classes=conf_matrix_labels, image_name='Conf_Matrix_Hierarchical_'+str(kfold_count),
                          normalize=True,
                          title='Confusion Matrix')

    kfold_count += 1

print('--------------Average results--------------')
print('Average Accuracy: {}'.format(np.mean(accuracy_array)))





