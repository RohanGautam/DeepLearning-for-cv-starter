from pyimagesearch.nn.neuralnetwork import NeuralNetwork
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets


def neuralNetTriainTest(X, y):
    nn = NeuralNetwork([2, 2, 1], alpha=0.5)
    nn.fit(X, y, epochs=20000)
    # now that our network is trained, loop over the XOR data points
    for (x, target) in zip(X, y):
        # make a prediction on the data point and display the result
        # to our console
        pred = nn.predict(x)[0][0]
        step = 1 if pred > 0.5 else 0  # step function
        print(
            f"[INFO] data={x}, ground-truth={target}, pred={pred}, step={step}")


# for XOR
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 1, 1, 0])
# print("\n\tXOR dataset:")
# neuralNetTriainTest(X, y)

# for MNIST, built into sklearn
# load the MNIST dataset and apply min/max scaling to scale the
# pixel intensity values to the range [0, 1] (each image is
# represented by an 8 x 8 = 64-dim feature vector)
print("[INFO] loading MNIST (sample) dataset...")
digits = datasets.load_digits()
data = digits.data.astype("float")
data = (data - data.min()) / (data.max() - data.min())
print("[INFO] samples: {}, dim: {}".format(data.shape[0],
                                           data.shape[1]))

# construct the training and testing splits
(trainX, testX, trainY, testY) = train_test_split(data,
digits.target, test_size=0.25)
# convert the labels(the Y's) from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# train the network
print("[INFO] training network...")
nn = NeuralNetwork([trainX.shape[1], 32, 16, 10]) # op 10 digits, hence 10 nodes in last network
print("[INFO] {}".format(nn))
nn.fit(trainX, trainY, epochs=1000)

# evaluate the network
print("[INFO] evaluating network...")
predictions = nn.predict(testX)
predictions = predictions.argmax(axis=1) # to selct classification with highest probability for each data point
print(classification_report(testY.argmax(axis=1), predictions))