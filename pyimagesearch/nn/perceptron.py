import numpy as np


class Perceptron:
    def __init__(self, N, alpha=0.1):
        # N+1 values from std normal(div by sqrt to scale it down), extra one for bias entry
        self.W = np.random.randn(N+1)/np.sqrt(N)
        self.alpha = alpha

    def step(self, x):
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs=10):
        '''Common to name training function fit,
        "fitting" a model to our data
        X-> data, y-> actual labels model should be predicting'''
        # to treat bias as trianable parameter
        X = np.hstack([
            X,
            np.ones((X.shape[0], 1))
        ])
        # the actual training procedure
        for epoch in np.arange(epochs):
            for (data, target) in zip(X, y):
                model_op = self.step(np.dot(data, self.W))
                # only update if prediction does not match target
                if(model_op != target):
                    error = model_op-target
                    self.W += -self.alpha*error*data

    def predict(self, X, addBias=True):
        #to ensure X is a matrix
        X = np.atleast_2d(X)
        if addBias:
            # to treat bias as trianable parameter
            X = np.hstack([
                X,
                np.ones((X.shape[0], 1))
            ])
        
        return self.step(np.dot(X, self.W))