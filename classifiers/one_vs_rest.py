"""
    One vs Rest class decorator. I use a decorator because some classes
    were already written and I didn't want to rewrite them.

    The decorator adds functionality to fit and predict's classifier functions
    to deal with multiclass labels.
"""

import numpy as np
import math


def max_labels(d):
    result = []
    for i in range(len(next(iter(d.values())))):
        max = -math.inf
        label = None
        for k,v in d.items():
            if v[i] > max:
                max = v[i]
                label = k
        result.append(label)
    return np.array(result)


class OneVsRest:
    def __init__(self, aklass):
        self.aklass = aklass
        self._instance = None

    def __call__(self, *args, **kwargs):
        if not self._instance:
            self._instance = self.aklass(*args, **kwargs)
            self.classifiers = {}
        return self

    def fit(self, X, y):

        labels = np.unique(y)
        self.dict_labels = dict(zip(labels, range(len(labels))))

        for label in labels:
            y_label = [1 if elem == label else -1 for elem in y]
            self.classifiers[self.dict_labels[label]] = self._instance.fit(X, y_label)._weights
        return self

    def predict(self, xi):
        values = {}
        for k, v in self.classifiers.items():
            self._instance._weights = v
            values[k] = self._instance.net_input(xi)

        return max_labels(values)