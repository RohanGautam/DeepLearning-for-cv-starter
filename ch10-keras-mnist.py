from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import argparse

plotPath = "outputs/output.png"
# grab the MNIST dataset (if this is your first time running this
# script, the download may take a minute -- the 55MB MNIST dataset
# will be downloaded)
print("[INFO] loading MNIST (full) dataset...")
X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
print("[INFO] finished loading MNIST dataset")


# scale the raw pixel intensities to the range [0, 1.0], then
# construct the training and testing splits
X = X.astype("float") / 255.0
(trainX, testX, trainY, testY) = train_test_split(X,
                                                  y, test_size=0.25)

# convert the labels from integers to vectors
# eg: '3' becomes [0,0,0,1,0,0,0,0,0,0] (vector with 1 at 3's index)
# man ml algorithms benefit from this
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# X.shape is (70k, 784) -> dimension of each sample is 784
# this architecture implies that the layers will be stacked on top of each other
model = Sequential()
model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
# softmax for probabilities for each class prediction
model.add(Dense(10, activation="softmax"))


# train the model using SGD
print("[INFO] training network...")
sgd = SGD(0.01)
# more model definitions
model.compile(loss="categorical_crossentropy", optimizer=sgd,
              metrics=["accuracy"])
# train/fit
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              epochs=100, batch_size=128)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=[str(x) for x in lb.classes_]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(plotPath)