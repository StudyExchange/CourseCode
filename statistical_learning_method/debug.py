import numpy as np
import math

class BaseClassifier(object):
    def __init__(self):
        self._error = None

    def fit(self, x_data, y_data):
        pass
    def predict(self, x_test):
        pass
    @property
    def error(self):
        return self._error

class Adaboost(object):
    def __init__(self, classifier_count = 10):
        self._x_train = None
        self._y_train = None
        self._classifier_count = classifier_count
        

    def fit(self, x_train: np.array, y_train: np.array):
        self._x_train = x_train
        self._y_train = y_train
        self._m = len(self._x_train)
        self._n = len(self._x_train[0])

        self._clf = None
        self._error = None
        # self._weight = None
        self._alpha = None

        self._classifiers = []
        self._errors = []
        self._weights = []
        self._alphas = []
        # 初始化权值
        self._weight = np.ones((self._m,)) / self._m
        for i in range(self._classifier_count):
            x = self._weight * self._x_train
            self._clf = BaseClassifier()
            self._clf.fit(x, self._y_train)
            self._error = self._clf.error
            self._alpha = 1. / 2 * math.log((1 - self._error) / self._error)
            self._weight = self._get_weight()

            self._classifiers.append(self._clf)
            self._errors.append(self._error)
            self._alphas.append(self._alpha)
            self._weights.append(self._weight)


    def predict(self, x_test):
        pass

    @property
    def x_fit(self):
        return self._x_train
    @property
    def y_fit(self):
        return self._y_train
    
    def _get_weight(self):
        alpha = - self._alpha
        weights = self._weight*np.log(alpha*self._y_train*self._clf(self._x_train))
        z = sum(weights)
        wi1 = weights / z
        return wi1

x_train = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
y_train = np.array([1, 1, 1,-1,-1,-1, 1, 1, 1,-1]).reshape(-1, 1)

ada = Adaboost()
ada.fit(x_train, y_train)
