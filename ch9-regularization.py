# import the necessary packages
# Contains implementation for stochastic gradient descent
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing import simplepreprocessor
from pyimagesearch.datasets import simpledatasetloader
from imutils import paths
import argparse

# parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
args = vars(ap.parse_args())
# grab the list of image paths
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# load images from disk and resize them
sp = simplepreprocessor.SimplePreprocessor(32, 32)
preprocessors = [sp]
sdl = simpledatasetloader.SimpleDatasetloader(preprocessors)
data, labels = sdl.load(imagePaths, verbose=500)
# data now has 3000 (32x32x3) images. We flatten each image, with data having 3000 rows and each row with a list of len 3072
data = data.reshape(3000, 32*32*3)

# encode the labels as integers
le = LabelEncoder()
# the original labels can be accesed by le.classes_
labels = le.fit_transform(labels)

# 75% train, 25% test
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, train_size=0.25, random_state=5)

# trying out a few different regularisation methods
for r in (None, "l1", "l2"):
    print(f"[INFO] using {r} regularisation/ penalty")
    # λ (regularisation strength) is 0.0001 (default)
    model = SGDClassifier(loss="log",  # "log" is cross-entropy loss
                          penalty=r,
                          max_iter=20,  # for 10 epochs
                          learning_rate="constant",
                          # initial learning rate. Will continue to be this, as learning rate is set constant here.
                          eta0=0.01,
                          random_state=42)

    model.fit(trainX, trainY)
    # evaluate the classifier
    acc = model.score(testX, testY)
    print("[INFO] ‘{}‘ penalty accuracy: {:.2f}%".format(r,
                                                         acc * 100))
