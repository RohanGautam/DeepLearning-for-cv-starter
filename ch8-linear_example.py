import numpy as np
import cv2
# initialize the class labels and set the seed of the pseudorandom
# number generator so we can reproduce our results
labels = ["dog", "cat", "panda"]
np.random.seed(1) # works because this is SET ON PURPOSE.
# randomly initialising weights and bias matrices.
# Would be "learned" and finetuned in an actual model
W = np.random.randn(3, 32*32*3) # W,b would be optimised via a gradient descent process
b= np.random.randn(3)

orig = cv2.imread("datasets/dog.png")
image=cv2.resize(orig, (32,32)).flatten()

scores = W.dot(image) + b

for(label, score) in zip(labels, scores):
    print(f'[INFO] {label} : {score}')

cv2.putText(orig, "Label: {}".format(labels[np.argmax(scores)]),
(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
# display our input image
cv2.imshow("Image", orig)
cv2.waitKey(0)