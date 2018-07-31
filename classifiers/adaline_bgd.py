import numpy as np


class AdalineBGD:
    def __init__(self, learning_rate, number_of_epochs, batch_size=32):
        self.learning_rate = learning_rate
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size

        self._weights = None
        self.cost_ = []

    def fit(self, X, y):
        input_size = X.shape[1] + 1
        self._weights = np.random.normal(loc=0.0, scale=0.01, size=input_size)

        for _ in range(self.number_of_epochs):

            batches_costs = []
            num_batches = 0
            for xi, yi in self.generate_batches(X, y):
                net_input = self.net_input(xi)
                output = self.activation(net_input)
                error = yi - output

                self._weights[1:] += self.learning_rate * xi.T.dot(error)
                self._weights[0] += self.learning_rate * error.sum()

                batch_cost = (error ** 2).sum() / 2.0
                num_batches += 1
                batches_costs.append(batch_cost)

            avg_cost = sum(batches_costs) / num_batches
            self.cost_.append(avg_cost)

        return self

    def net_input(self, X):
        return np.dot(X, self._weights[1:]) + self._weights[0]

    def activation(self, X):
        return X

    def predict(self, xi):
        return np.where(self.activation(self.net_input(xi)) >= 0, 1, -1)

    def generate_batches(self, X, y):
        for i in range(X.shape[0] // self.batch_size):
            start = i*self.batch_size
            end = (i+1)*self.batch_size
            yield (X[start:end], y[start:end])

        if X.shape[0] % self.batch_size != 0:
            start = (i+1)*self.batch_size
            end = X.shape[0]+1
            yield (X[start:end], y[start:end])