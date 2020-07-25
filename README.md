# covid-sp


This is a project to implement an image feature extractor and a binary classifier.
First, clone this project and import it in your IDE.

## Instructions to run the feature extraction:

This project contains a LBP (Local Binary Pattern) feature extractor. There were two implementations provided: one using 
uniform method and another using nri_uniform method. The first was provided by Rodolfo and the second by Aguiar.

1. First, open the lbp_feature_extraction.py script and look at the following variables:

    1.1 train_directory variable is used to point to the directory where the training dataset images will be. 
    Inside that directory, there are sub-directories which will represent each class 
    of the dataset. It is important to set the sub-directory name as the desired name for each class once 
    that this will be used to generate the final csv file. It is also important to have the correct images in the 
    correspondent directories (If an image is of class c1, then it should be put inside sub-directory c1). 
    
    1.2 test_directory variable is used to point where the testing dataset images will be.
    Inside that directory, there are only images of the training set, once we don't know to which class they belong.
    The name of the images will be used as ids for them.
    
    1.3 lbp_extractor is a variable to control which feature extractor will be used. If it 'uniform' is passed, Rodolfo's implementation will be used. 
    Otherwise, if 'nri-uniform' is passed, Aguiar's implementation will be used. The first has 10 features (columns) and the second 60.
    
2. After making sure we are using the correct parameters, run the lpb_feature_extraction.py script.

3. After running, check in your project the 'Feature Matrix Train' and 'Feature Matrix Test' directories. The first will have a csv that will have the training dataset, with the 
image class in the last column. The other one will have the testing dataset, with the id of the image in the last column.


## Instructions to run the classifier

1. Open the classifier.py script and look at the following variables:

    1.1 train_data is a variable to point to the folder where the csv file with the training dataset was saved
    
    1.2 test_data is a variable to point to the folder where the csv file with the testing dataset was saved
    
    1.3 classifier is a variable to set which type of classifier we will use: 'rf' for Random Forest, 'mlp' for Multilayer Perceptron, 'svm' for SVM.
    
2. After making sure we are using the correct parameters, run the classifier.py script.

3. After running, check in your project the 'Results' directory. It will be created a csv file with the results. The format is the following:

    id_exame,class,prob


