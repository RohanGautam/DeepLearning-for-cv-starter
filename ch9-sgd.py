from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100,
                help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01,
                help="learning rate")
ap.add_argument("-b", "--batch-size", type=int, default=32,
                help="size of SGD mini-batches")
args = vars(ap.parse_args())

'''
SGD- stochastic gradient descent
    > Foundation for modern-day machine learning
    > Gradient calculation and weight updation for every *batch*
    of data, not for every data point every epoch like vanilla gradient descent
'''


def sigmoid_activation(x):
    '''
    > compute the sigmoid activation value for a given input
    > sigmoid will give a value between 0 and 1. very negative->close to 0, very pos->close to 1
    > if the result of sigmoid is closer to 1, it's more activated.
    '''
    return 1/(1+np.exp(-x))  # np.exp so that it can apply the operation to x even if it is a matrix


def next_batch(X, y, batch_size):

    for i in np.arange(0, X.shape[0], batch_size):
        yield (X[i: i+batch_size], y[i: i+batch_size])


def predict(X, W):
    '''
    `X` is our design matrix, each row being a data point(flattened image in this case)
    `W` is our matrix of weights, with the last column being just `1`'s, so that we can use the bias trick,
    treating the bias vector as another tune-able parameter.
    '''

    preds = sigmoid_activation(X.dot(W))
    # any prediction >0.5 as 1, <=0.5 as 0. [because sigmoid is centered at 0.5]
    # done to threshold the outputs to binary class labels
    # we can make this "thresholding"/classification only after "activating" the values, then seeing
    # "how active"[close to 1] they are
    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1

    return preds


# 2-class classification problem. 1000 data points with each being a 2d feature vector
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2,
                    cluster_std=1.5, random_state=1)
# convert Y from a 1d array with 1000 elements, to a 2d matrix with 1000 rows,
# each prev ele being in a row of it's own.
# The labels for each of these data points are either 0 or 1.
y = y.reshape((y.shape[0], 1))

# Adding a column of 1's to the right of X (?)
X = np.hstack([
    X,
    np.ones((X.shape[0], 1))
])
# partition, using 50% for triaining and 50% for testing
# X,y from above are being used as data and labels respectively
(trainX, testX, trainY, testY) = train_test_split(X, y,
                                                  test_size=0.5, random_state=42)

print("[INFO] training...")
W = np.random.randn(X.shape[1], 1)
losses = []

for epoch in np.arange(0, args["epochs"]):
    epochLoss = []

    for (batchX, batchY) in next_batch(trainX, trainY, args["batch_size"]):
        preds = sigmoid_activation(batchX.dot(W))
        error = preds - batchY
        epochLoss.append(np.sum(error ** 2))

        gradient = batchX.T.dot(error)
        W += -args["alpha"] * gradient
    
    loss = np.average(epochLoss)
    losses.append(loss)

    # check to see if an update should be displayed
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1),
                                                    loss))

# evaluate our model
print("[INFO] evaluating...")
preds = predict(testX, W)
print(classification_report(testY, preds))

# plot the (testing) classification data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", s=30)

# construct a figure that plots the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
