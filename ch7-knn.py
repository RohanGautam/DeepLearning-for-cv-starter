# run : python ch7-knn.py -d datasets/animals

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from pyimagesearch.preprocessing import simplepreprocessor
from pyimagesearch.datasets import simpledatasetloader
from imutils import paths
import argparse

'''
K-NN:
    Classifies unknown input to most common class among 'k'-nearest
    data points in a n-dim space.
    Have to choose: 
        'k': for it to choose the k-closest data points
        distance metric: used by our definition of "close". Eg: euclidean dist, manhattan dist, etc.
'''


'''Adding the required command line arguments'''
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
                help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
                help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())  # eq. to ap.parse_args().__dict__

'''Step 1: getting the dataset'''
print("[INFO] loading images...")
dataPath = args["dataset"]
# get the full image paths, using a convinence function from imutils
paths = list(paths.list_images(dataPath))
# initialize the preprocessor, which for us, resizes img w/o aspect ratio consideration
preprocessor = simplepreprocessor.SimplePreprocessor(32, 32) # 32x32 size
# get the data from the dataset, with preprocessor applied to it
sdl = simpledatasetloader.SimpleDatasetloader([preprocessor])
(data, labels) = sdl.load(paths, verbose=500)
# to apply the knn alg, we need to flatten images from 32x32x3 to 3072(32*32*3)
data = data.reshape(data.shape[0], 32*32*3) # shape will change from (3000,32,32,3) to (3000, 3072)
# show some information on memory consumption of the images
print(f"[INFO] features matrix: {data.nbytes / (1024 * 1000.0)}MB")

'''Step 2: splitting dataset into train, test, and validation sets'''
# converting labels like cat, dog etc to numbers: 0,1,2 (encode the labels as integers)
le = LabelEncoder()
labels = le.fit_transform(labels) # set(labels) = {0,1,2}

#splitting. Train-75%, test:25%. random_state sets the seed for random division
# trainX, testX: train and test data, trainY, testY: labels
# X refers to data, Y refers to labels
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size = 0.25, random_state = 42)

'''Step 3: building and training network'''
# uses euclidean distance by default
# train and evaluate a k-NN classifier on the raw pixel intensities
print("[INFO]training and evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"], 
                             n_jobs=args["jobs"])

model.fit(trainX, trainY) # pass train data and labels

'''Step 4: evaluate with test set'''
print(classification_report(testY, model.predict(testX), target_names=le.classes_)) #le.classes_is the orignial label names
