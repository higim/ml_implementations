import numpy as np


class Perceptron:
    """
        Implement a perceptron from scratch, only for learning purposes.

        Algorithm which can learn automatically the optimal weights for
        the coefficients, whose are, then, multiplied for the input
        to decide if the neuron is shot or not.
    """

    def __init__(self, learning_rate, number_of_epochs):
        self._learning_rate = learning_rate
        self._number_of_epochs = number_of_epochs

        self._weights = None
        self._errors = []

    def fit(self, X, y):
        """
        Steps:
        1. Initialize w's to small random numbers.
        2. For each x:
            a. Calculate ŷ -> Class label predicted by the function.
            b. Upload w's.


        :param X: Input, type n-array, column vector.
        :param y: Real class labels
        :return: self
        """
        input_size = X.shape[1] + 1
        self._weights = np.random.normal(loc=0.0, scale=0.01, size=input_size)

        for _ in range(self._number_of_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                predicted_class_label = self.predict(xi)
                update = self._learning_rate * (yi - predicted_class_label)
                self._weights[1:] += update * xi
                self._weights[0] += update

                if predicted_class_label != yi:
                    errors += 1
            self._errors.append(errors)

        return self

    def net_input(self, X):
        return np.dot(X, self._weights[1:]) + self._weights[0]

    def predict(self, xi):
        return np.where(self.net_input(xi) >= 0, 1, -1)