from pyimagesearch.nn.perceptron import Perceptron
import numpy as np


def perceptronTrainTest(X, y):
    # training the perceptron. N is the number of data points
    print("[INFO] training perceptron...")
    p = Perceptron(X.shape[1], alpha=0.1)
    p.fit(X, y, epochs=20)
    # evaluating
    print("[INFO] testing perceptron...")
    for (x, target) in zip(X, y):
        print(f'Input: {x}, expected: {target}, predicted: {p.predict(x)}')


X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
# Output for OR dataset
y = np.array([0, 1, 1, 1])
print("\n\tOR dataset:")
perceptronTrainTest(X, y)
# Output for AND dataset
y = np.array([0, 0, 0, 1])
print("\n\tAND dataset:")
perceptronTrainTest(X, y)
# Output for XOR dataset
y = np.array([0, 1, 1, 0])
print("\n\tXOR dataset:")
perceptronTrainTest(X, y)

'''
As we see, it correctly models the OR and AND dataset as they are linearly seperable.
However, it fails to do so for the XOR dataset which is *nonlinearly* seperable.
'''
