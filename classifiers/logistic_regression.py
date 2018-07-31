import numpy as np


class LogisticRegressionGD:
    def __init__(self, learning_rate, number_of_epochs, random_state=1):
        self.learning_rate = learning_rate
        self.number_of_epochs = number_of_epochs
        self.random_state = random_state

        self._weights = None
        self.cost_ = []

    def fit(self, X, y):

        input_size = X.shape[1] + 1 # +1 for bias weight w0

        rgen = np.random.RandomState(self.random_state)
        self._weights = rgen.normal(loc=0.0, scale=0.01, size=input_size)

        for _ in range(self.number_of_epochs):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            error = y - output

            # Update weights
            self._weights[1:] += self.learning_rate * X.T.dot(error)
            self._weights[0] += self.learning_rate * error.sum()

            cost = (-y.dot(np.log(output)) - ((1-y).dot(np.log(1-output))))
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        return np.dot(X, self._weights[1:]) + self._weights[0]

    def activation(self, z):
        return self.sigmoid(z)

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def predict(self, xi):
        return np.where(self.activation(self.net_input(xi)) >= 0.5, 1, 0)


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from utils.ploting import plot_decision_regions
    import matplotlib.pyplot as plt

    # Check if Logistic Regression Implementation works
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    # Split the dataset into separate training and test datasets
    # test_size: %'s size for the test dataset
    # random_state = fixed random seed for shuffling datasets prior to splitting
    # stratify:y training and test subsets have the same proportions of class labels as the "y".
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    # Get only Iris-setosa and Iris-versicolor because the lrgd only works for binary classifications
    X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
    y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
    lrgd = LogisticRegressionGD(learning_rate=0.05, number_of_epochs=100, random_state=1)
    lrgd.fit(X_train_01_subset, y_train_01_subset)
    plot_decision_regions(X=X_train_01_subset,
                          y=y_train_01_subset,
                          classifier=lrgd)

    plt.xlabel('petal length [std]')
    plt.ylabel('petal width [std]')
    plt.legend(loc='upper left')
    plt.show()
