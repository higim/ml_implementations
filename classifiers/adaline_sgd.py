import numpy as np
from sklearn.utils import shuffle


class AdalineSGD:
    def __init__(self, learning_rate, number_of_epochs):
        self.learning_rate = learning_rate
        self.number_of_epochs = number_of_epochs

        self._weights = None
        self.cost_ = []

    def fit(self, X, y):

        self.initialize_weights(X.shape[1])
        for _ in range(self.number_of_epochs):

            # Shuffle data for Stochastic
            # Another way to shuffle is using numpy.Random.permutation
            # r = np.random.permutation(len(y))
            # X[r], y[r]

            X_s, y_s = shuffle(X,y)
            cost = []
            for xi, yi in zip(X_s, y_s):
                cost.append(self.update_weights(xi,yi))

            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)

        return self

    def partial_fit(self, X, y):
        """ Train with new samples on the fly. """

        if self._weights is None:
            self.initialize_weights(X.shape[1])

        if y.ravel().shape[0] > 1:
            for xi, yi in zip(X, y):
                self.update_weights(xi,yi)

        else:
            self.update_weights(X,y)

        return self

    def initialize_weights(self, X_size):
        """ A separate function is necessary because weights can be initialized in fit or partial fit"""
        self._weights = np.random.normal(loc=0.0, scale=0.01, size=X_size+1)

    def update_weights(self, xi, yi):
        net_input = self.net_input(xi)
        output = self.activation(net_input)
        error = yi - output

        self._weights[1:] += self.learning_rate * xi.T.dot(error)
        self._weights[0] += self.learning_rate * error

        cost = 0.5*error**2
        return cost

    def net_input(self, X):
        return np.dot(X, self._weights[1:]) + self._weights[0]

    def activation(self, X):
        return X

    def predict(self, xi):
        return np.where(self.activation(self.net_input(xi)) >= 0, 1, -1)