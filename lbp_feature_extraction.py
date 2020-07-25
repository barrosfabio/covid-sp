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
def feature_extraction(image):
    lbp = LocalBinaryPatterns(8, 2)
    image_matrix = np.array(image.convert('L'))
    img_features = lbp.describe_lbp_method_rd(image_matrix)

    return img_features.tolist()

def create_columns(column_number):
    columns = []
    for i in range(0, column_number):
        columns.append(str(i))

    columns.append('class')
    return columns


# Setting up the train directory
train_directory = 'Train'

# Creating a dataframe to store all the features
columns = create_columns(10)
data_frame = pd.DataFrame(columns=columns)

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

        img_features = feature_extraction(open_img(file_path))

        # The name of the directory is the class
        img_features.append(dir)

        rows_list.append(img_features)

# Creating a dataframe to store all the features
feature_matrix = pd.DataFrame(rows_list,  columns= columns)

#Write features to csv
feature_matrix.to_csv('feature_matrix_train.csv', index=False)





