import numpy as np
from classifiers.one_vs_rest import OneVsRest

@OneVsRest
class AdalineGD:
    def __init__(self, learning_rate, number_of_epochs):
        self.learning_rate = learning_rate
        self.number_of_epochs = number_of_epochs

        self._weights = None
        self.cost_ = []
        self.errors = []

    def fit(self, X, y):
        """
        Steps:
            1. Initialize w's to small random numbers.
            2. For each x:
                a. Calculate activation -> Linear function.
                b. Upload w's / Minimize costs.

            :param X: Input, type n-array, column vector.
            :param y: Real class labels
            :return: self
        """

         # Initialize weights
        input_size = X.shape[1] + 1
        self._weights = np.random.normal(loc=0.0, scale=0.01, size=input_size)

        for _ in range(self.number_of_epochs):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output

            self._weights[1:] += self.learning_rate * X.T.dot(errors)
            self._weights[0] += self.learning_rate * errors.sum()

            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        return np.dot(X, self._weights[1:]) + self._weights[0]

    def activation(self, X):
        # Activation function in ADAline is identity function.
        # We could omit this, but we maintain it
        # for math clarity and to preserve the scheme
        # net input -> activation -> predict
        return X

    def predict(self, xi):
        return np.where(self.activation(self.net_input(xi)) >= 0, 1, -1)