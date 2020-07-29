import numpy as np
from skimage.feature import local_binary_pattern
from PIL import Image
import os
import pandas as pd
import imghdr

UNIFORM_FEATURE_NUMBER = 10
NRI_UNIFORM_FEATURE_NUMBER = 59

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

# Verify if a given image is using a valid format
def verify_valid_img(path):
    possible_formats = ['png','jpg','jpeg','tiff','bmp','gif']
    if imghdr.what(path) in possible_formats:
        return True
    else:
        return False

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

    print("Started feature extraction for the training dataset")

    # Iterate over subdirectories in training folder (1 folder for each class)
    for dir in os.listdir(train_directory):

        # This is the path to each subdirectory
        sub_directory = train_directory + '\\' + dir

        # Retrieve the files for the given subdirectory
        training_filelist = os.listdir(sub_directory)

        # Iterate over all the files in the class folder
        for file in training_filelist:
            file_path = sub_directory + '\\' + file

            if verify_valid_img(file_path):
                print("Processing: "+file_path)

                image = open_img(file_path)
                img_features = feature_extraction(image, lbp_extractor)

                # The name of the directory is the class
                img_features.append(dir)

                rows_list.append(img_features)
            else:
                print("The following file is not a valid image: "+file_path)

    # Creating a dataframe to store all the features
    if lbp_extractor == 'uniform':
        columns = create_columns(UNIFORM_FEATURE_NUMBER,'class')
    elif lbp_extractor == 'nri_uniform':
        columns = create_columns(NRI_UNIFORM_FEATURE_NUMBER,'class')

    feature_matrix = pd.DataFrame(rows_list, columns=columns)

    print("Finished creating Training Feature Matrix")

    return feature_matrix

# Function to create the testing feature matrix, it has the id of each sample instead of the class
def create_feature_matrix_test(test_directory, lbp_extractor):
    # Variable to store the data_rows
    rows_list = []

    print("Started feature extraction for the training dataset")

    # Retrieve list of files in test directory
    test_filelist = os.listdir(test_directory)

    # Iterate over all the files in the class folder
    for file in test_filelist:
        file_path = test_directory + '\\' + file

        if verify_valid_img(file_path):

            print("Processing: " + file_path)

            image = open_img(file_path)
            img_features = feature_extraction(image, lbp_extractor)

            # id of the sample
            img_features.append(file[:-4])

            rows_list.append(img_features)
        else:
            print("The following file is not a valid image: " + file_path)


    # Creating a dataframe to store all the features
    if lbp_extractor == 'uniform':
        columns = create_columns(UNIFORM_FEATURE_NUMBER,'id_exame')
    elif lbp_extractor == 'nri_uniform':
        columns = create_columns(NRI_UNIFORM_FEATURE_NUMBER,'id_exame')

    feature_matrix = pd.DataFrame(rows_list, columns=columns)

    print("Finished creating Testing Feature Matrix")

    return feature_matrix


# Setting up the train and test directories
train_directory = 'T:\Projetos_Documentos\GitHub\covid-sp\Train'
test_directory = 'T:\Projetos_Documentos\GitHub\covid-sp\Test'
lbp_extractor = 'nri_uniform'

# Setting up the resulting matrices directories
feature_matrix_train_path = 'Feature Matrix Train'
feature_matrix_test_path = 'Feature Matrix Test'

if not os.path.isdir(feature_matrix_train_path):
    print('Creating Directory: '+feature_matrix_train_path)
    os.mkdir(feature_matrix_train_path)

if not os.path.isdir(feature_matrix_test_path):
    print('Creating Directory: '+feature_matrix_test_path)
    os.mkdir(feature_matrix_test_path)

feature_matrix_train = create_feature_matrix_train(train_directory, lbp_extractor)
print("Saving Training Feature Matrix to CSV")
feature_matrix_train.to_csv(feature_matrix_train_path + '\\feature_matrix_train.csv', index=False)
feature_matrix_test = create_feature_matrix_test(test_directory, lbp_extractor)
print("Saving Test Feature Matrix to CSV")
feature_matrix_test.to_csv(feature_matrix_test_path + '\\feature_matrix_test.csv', index=False)







