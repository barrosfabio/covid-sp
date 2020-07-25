import numpy as np
from skimage.feature import local_binary_pattern
from PIL import Image
import os
import pandas as pd

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    # LBP Feature Extractor from Rodolfo
    def describe_lbp_method_rd(self, image, eps=1e-7):
        lbp = local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))

        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        return hist

    # LBP Feature Extractor from Aguiar
    def describe_lbp_method_ag(self, image):
        lbpU = local_binary_pattern(image, self.numPoints, self.radius, method='nri_uniform')
        hist0, nbins0 = np.histogram(np.uint8(lbpU), bins=range(60), normed=True)

        return hist0

# Function to load an image from a path
def open_img(filename):
    img = Image.open(filename)
    return img

# Feature extraction call
def feature_extraction(image, lbp_extractor):
    lbp = LocalBinaryPatterns(8, 2)
    image_matrix = np.array(image.convert('L'))

    if(lbp_extractor == 'uniform'):
        img_features = lbp.describe_lbp_method_rd(image_matrix)
    elif(lbp_extractor == 'nri_uniform'):
        img_features = lbp.describe_lbp_method_ag(image_matrix)


    return img_features.tolist()

def create_columns(column_number, property):
    columns = []
    for i in range(0, column_number):
        columns.append(str(i))

    columns.append(property)
    return columns

# Function to create the training feature matrix, it has the expected class for each sample
def create_feature_matrix_train(train_directory, lbp_extractor):
    # Variable to store the data_rows
    rows_list = []

    # Iterate over subdirectories in training folder (1 folder for each class)
    for dir in os.listdir(train_directory):

        # This is the path to each subdirectory
        sub_directory = train_directory + '\\' + dir

        # Retrieve the files for the given subdirectory
        training_filelist = os.listdir(sub_directory)

        # Iterate over all the files in the class folder
        for image in training_filelist:
            file_path = sub_directory + '\\' + image

            img_features = feature_extraction(open_img(file_path), lbp_extractor)

            # The name of the directory is the class
            img_features.append(dir)

            rows_list.append(img_features)

    # Creating a dataframe to store all the features
    if(lbp_extractor == 'uniform'):
        columns = create_columns(10,'class')
    elif(lbp_extractor == 'nri_uniform'):
        columns = create_columns(60,'class')

    feature_matrix = pd.DataFrame(rows_list, columns=columns)

    return feature_matrix

# Function to create the testing feature matrix, it has the id of each sample instead of the class
def create_feature_matrix_test(test_directory, lbp_extractor):
    # Variable to store the data_rows
    rows_list = []

    # Retrieve list of files in test directory
    test_filelist = os.listdir(test_directory)

    # Iterate over all the files in the class folder
    for image in test_filelist:
        file_path = test_directory + '\\' + image

        img_features = feature_extraction(open_img(file_path), lbp_extractor)

        # id of the sample
        img_features.append(image[:-4])

        rows_list.append(img_features)

    # Creating a dataframe to store all the features
    if(lbp_extractor == 'uniform'):
        columns = create_columns(10,'id_exame')
    elif(lbp_extractor == 'nri_uniform'):
        columns = create_columns(60,'id_exame')

    feature_matrix = pd.DataFrame(rows_list, columns=columns)

    return feature_matrix



# Setting up the train and test directories
train_directory = 'Train'
test_directory = 'Test'
lbp_extractor = 'uniform'

feature_matrix_train = create_feature_matrix_train(train_directory, lbp_extractor)
feature_matrix_test = create_feature_matrix_test(test_directory, lbp_extractor)

# Setting up the resulting matrices directories
feature_matrix_train_path = 'Feature Matrix Train'
feature_matrix_test_path = 'Feature Matrix Test'

if not os.path.isdir(feature_matrix_train_path):
    os.mkdir(feature_matrix_train_path)

if not os.path.isdir(feature_matrix_test_path):
    os.mkdir(feature_matrix_test_path)

#Write features to csv
feature_matrix_train.to_csv(feature_matrix_train_path + '\\feature_matrix_train.csv', index=False)
feature_matrix_test.to_csv(feature_matrix_test_path + '\\feature_matrix_test.csv', index=False)





