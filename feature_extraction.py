import numpy as np
from skimage.feature import local_binary_pattern
from PIL import Image


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def describe_lbp_method1(self, image, eps=1e-7):
        lbp = local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))

        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        return hist

    def describe_lbp_method2(self, image):
        lbpU = local_binary_pattern(image, self.numPoints, self.radius, 'non-uniform')
        hist0, nbins0 = np.histogram(np.uint8(lbpU), bins=range(60), normed=True)

        return hist0

def open_img(filename):
    img = Image.open(filename)
    return img

def feature_extraction(image):
    lbp = LocalBinaryPatterns(8, 2)
    image_matrix = np.array(image.convert('L'))
    img_features = lbp.describe_lbp_method2(image_matrix)

    return img_features


# Open the image file
filename = 'NYNight.jpg'

# Extract the features
img_features = feature_extraction(open_img(filename))

#Verify what are the image features
print(img_features)

#Write features to csv





