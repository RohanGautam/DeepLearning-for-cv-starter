# Cifar-10 : 60k rgb images
# 10 classes : airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse
# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-o", "--output", required=True,
                # help="path to the output loss/accuracy plot")
# args = vars(ap.parse_args())
output = "outputs/keras-cifar10.png"

# %%
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
# convert data to float, and bring it to the range [0,1]
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0
# reshape the data (32*32*3 = 3072). Flatten this image into a single list of floating point values
# We have 50k train and 10k test
trainX = trainX.reshape((trainX.shape[0], 3072))
testX = testX.reshape((testX.shape[0], 3072))

# %%
# convert the labels from integers to vectors
lb = LabelBinarizer()    
# Ex: [1,2,1,2,1,3] -> labels: [1,2,3]
# We first need to transform these integer labels into vector labels, where the index in
# the vector for label is set to 1 and 0 otherwise (this process is called one-hot encoding).
trainY = lb.fit_transform(trainY)
# assign labels for test set too. Recap: `Y` means the labels
testY = lb.transform(testY)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

# %%
# define the 3072-1024-512-10 architecture using Keras
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="relu"))  # sigmoid is old
model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation="softmax"))

# %%
# Now that we have architecture defined, we can train it with schotastic gradient descent.
sgd = SGD(0.01)  # learning rate
model.compile(loss="categorical_crossentropy", optimizer=sgd,
              metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              epochs=100, batch_size=32)

# %%
# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1), target_names=labelNames))


# %%
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(output)

