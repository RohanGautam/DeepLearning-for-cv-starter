import numpy as np


class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.W = []  # weights
        # eg: if   [2,2,1]=> 2 input nodes, 2 hidden, 1 output
        self.layers = layers
        self.alpha = alpha  # learning rate
        # initializing weight matrix to connect every node in curent layer to every node in next layer
        for i in np.arange(0, len(layers)-2):
            # eg: [2,2,1] -> we'll have a 2x2 matrix to connect every node in L1 to L2
            # +1's @ end for bias term
            w = np.random.randn(layers[i]+1, layers[i+1]+1)
            # dividing by the square root of the number of nodes in the current layer
            self.W.append(w/np.sqrt(layers[i]))
        # the last two layers are a special case where the input
        # connections need a bias term but the output does not (note no +1 for the last layer,
        # as it doesnt need bias)
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        return f'NeuralNetwork : {"-".join(map(str,self.layers))}'

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # the derivative of the sigmoid function ASSUMING
        # that ‘x‘ has already been passed through the ‘sigmoid‘
        # function
        return x*(1-x)

    def fit(self, X, y, epochs=1000, displayUpdate=100):
        # insert a column of 1’s as the last entry in the feature
        # matrix -- this little trick allows us to treat the bias
        # as a trainable parameter within the weight matrix
        X = np.c_[X, np.ones((X.shape[0]))]

        # loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
            # loop over each individual data point and train
            # our network on it
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)
            # check to see if we should display a training update
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(
                    epoch + 1, loss))

    def fit_partial(self, x, y):
        # x-> individual data point, y-> label
        # construct list of activations as data flows through network.
        # first one is the data itself
        # to view inputs as arrays with at least two dimensions
        A = [np.atleast_2d(x)]

        # FEEDFORWARD-> move ahead, taking dot products on the way
        for layer in np.arange(0, len(self.W)):
            # net input: dot product of matrix and weight at that layer
            net = A[layer].dot(self.W[layer])
            # net output is only once activations are applied
            out = self.sigmoid(net)
            # add output to list of activations
            A.append(out)

        # BACKPROPAGATION
        # 1. calculate error in prediction(is in the last layer)
        error = A[-1]-y
        # 2. Apply chain rule -> build a list of deltas
        # initialise D with deltas of last layer
        D = [error*self.sigmoid_derivative(A[-1])]
        # going backwards from second last layer
        for layer in np.arange(len(A) - 2, 0, -1):
            # the delta for the current layer is equal to the delta
            # of the *previous layer* dotted with the weight matrix
            # of the current layer, followed by multiplying the delta
            # by the derivative of the nonlinear activation function
            # for the activations of the current layer
            delta = D[-1].dot(self.W[layer].T)
            delta *= self.sigmoid_derivative(A[layer])
            D.append(delta)
        # so that last layer's stuff is at the last
        D=D[::-1]

        # WEIGHT UPDATE
        for layer in np.arange(0, len(self.W)):
            # update our weights by taking the dot product of the layer
            # activations with their respective deltas,
            # multiplied by learning rate 
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])
    
    def predict(self, X, addBias=True):
        # initialize the output prediction as the input features -- this
        # value will be (forward) propagated through the network to
        # obtain the final prediction
        p = np.atleast_2d(X)
        # check to see if the bias column should be added
        if addBias:
            # insert a column of 1’s as the last entry in the feature
            # matrix (bias)
            p = np.c_[p, np.ones((p.shape[0]))]
        # loop over our layers in the network
        for layer in np.arange(0, len(self.W)):
            # computing the output prediction is as simple as taking
            # the dot product between the current activation value ‘p‘
            # and the weight matrix associated with the current layer,
            # then passing this value through a nonlinear activation
            # function
            p = self.sigmoid(np.dot(p, self.W[layer]))
        # return the predicted value, will have dimension of last layer
        # due to all the dot-productting 
        return p
        
    def calculate_loss(self, X, targets):
        # make predictions for the input data points then compute
        # the loss
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)
        # return the loss
        return loss