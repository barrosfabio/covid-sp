import numpy as np
from skimage.feature import local_binary_pattern
from PIL import Image
import os
import pandas as pd


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def describe_lbp_method_rd(self, image, eps=1e-7):
        lbp = local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))

        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        return hist

    def describe_lbp_method_ag(self, image):
        lbpU = local_binary_pattern(image, self.numPoints, self.radius, method='nri_uniform')
        hist0, nbins0 = np.histogram(np.uint8(lbpU), bins=range(60), normed=True)

        return hist0

def open_img(filename):
    img = Image.open(filename)
    return img

def feature_extraction(image):
    lbp = LocalBinaryPatterns(8, 2)
    image_matrix = np.array(image.convert('L'))
    img_features = lbp.describe_lbp_method_rd(image_matrix)

    return img_features.tolist()

# Open the image file
train_directory = 'Train'
training_filelist = os.listdir(train_directory)

# Creating a dataframe to store all the features

columns = ['1','2','3','4','5','6','7','8','9','10','class']
data_frame = pd.DataFrame(columns=columns)

rows_list = []

# Iterate over all the files in the Training folder
for image in training_filelist:

    file_path = train_directory + '\\' + image

    img_features = feature_extraction(open_img(file_path))
    img_features.append(image)

    rows_list.append(img_features)

# Creating a dataframe to store all the features
feature_matrix = pd.DataFrame(rows_list,  columns= columns)


#Write features to csv
feature_matrix.to_csv('feature_matrix.csv', index=False)





