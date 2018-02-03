import numpy as np
import math

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

class BaseClassifier(object):
    def __init__(self):
        self._split = 0
        self._error = 0
        self._less_than = True

    def fit(self, x_data: np.array, y_data: np.array, sample_weight=None):
        assert len(x_data.shape) == 1
        assert len(y_data.shape) == 1
        length = len(x_data)
        if sample_weight is None:
            sample_weight = np.ones((length, ))
        # 下面代码分了两段，需要比较到底并决定是用小于还是大于
        # 这里是为了用一个函数适应书中的G1(x)，G2(x)，G3(x)3个函数
        # 其中G1(x)和G2(x)是用的小于，G3(x)用的大于
        # 小于部分
        errors_lt = np.zeros((length,))
        for i in range(length):
            y_pre = np.zeros((length,))
            error_i = np.zeros((length,))
            for j in range(length):
                y_pre[j] = self._g_less_than(x_data[j], x_data[i])
            error_i = y_data != y_pre
            error_i = error_i.astype(int)
            # 【p138，公式8.1】
            errors_lt[i] = sum(error_i * sample_weight)
        split_index_lt = np.argmin(errors_lt)
        errors_gt = np.zeros((length,))
        # 大于部分
        for i in range(length):
            y_pre = np.zeros((length,))
            error_i = np.zeros((length,))
            for j in range(length):
                y_pre[j] = self._g_greater_than(x_data[j], x_data[i])
            error_i = y_data != y_pre
            error_i = error_i.astype(int)
            # 【p138，公式8.1】
            errors_gt[i] = sum(error_i * sample_weight)
        split_index_gt = np.argmin(errors_gt)
        # 比较小于和大于两个部分
        if errors_lt[split_index_lt] < errors_gt[split_index_gt]:
            self._split = x_data[split_index_lt]
            self._error = errors_lt[split_index_lt]
        else:
            self._less_than = False
            self._split = x_data[split_index_gt]
            self._error = errors_gt[split_index_gt]


    def predict(self, x_test):
        assert len(x_test.shape) == 1
        length = len(x_test)
        y_pre = np.zeros((length,))
        
        if self._less_than:
            for i in range(length):
                y_pre[i] = self._g_less_than(x_test[i], self._split)
        else:
            for i in range(length):
                y_pre[i] = self._g_greater_than(x_test[i], self._split)
        return y_pre

    # 【例8.1中，G1(x)，G2(x)是小于】
    def _g_less_than(self, x, split):
        return 1 if x < split else -1

    # 【例8.1中，G3(x)是大于】
    def _g_greater_than(self, x, split):
        return 1 if x > split else -1

    @property
    def error(self):
        return self._error

class Adaboost(object):
    def __init__(self, classifier_count = 10):
        self._x_train = None
        self._y_train = None
        # 【p138，算法8.1，第（2）步】这里M是分类器的数量，【140】中部，有提到
        # “步骤（3）线性组合f(x)实现M个基本分类器的加权表决”
        self._classifier_count = classifier_count
        

    def fit(self, x_train: np.array, y_train: np.array):
        self._x_train = x_train
        self._y_train = y_train
        self._m = len(self._x_train)
        # self._n = len(self._x_train[0])

        self._clf = None
        self._error = 0
        # self._weight = None
        self._alpha = 0

        self._classifiers = []
        self._errors = []
        self._weights = []
        self._alphas = []
        # 【p138，第（1）步，初始化权值】
        self._weight = np.ones((self._m,)) / self._m
        for i in range(self._classifier_count):
            self._clf = BaseClassifier()
            self._clf.fit(self._x_train, self._y_train, self._weight)
            self._error = self._clf.error
            # 【p139，公式8.2】
            self._alpha = 1. / 2 * math.log((1 - self._error) / self._error)
            self._weight = self._get_weight()

            self._classifiers.append(self._clf)
            self._errors.append(self._error)
            self._alphas.append(self._alpha)
            self._weights.append(self._weight)
            print('error: %s' % self._error)
            print('alpha: %s' % self._alpha)
            print('weight:%s' % self._weight)
            print('*'*30)
            # 终止条件，没有找到书中对应的内容，属于个人添加
            if(all(self.predict(x_train) == y_train)):
                print('全部正确分类，满足终止条件：%s of %s' %(i, self._classifier_count))
                break
            pass


    def predict_prob(self, x_test):
        y_preds = []
        for i, clf in enumerate(self._classifiers):
            y_preds.append(self._alphas[i] * clf.predict(x_test))
        y_pred = sum(y_preds)
        return y_pred
    def predict(self, x_test):
        # 【p8.7，公式8.7】这里把probability和sign分开成两个函数来实现，
        # 便于需要probability的情况
        y_preds = self.predict_prob(x_test)
        result = np.ones((len(y_preds),))
        for i in range(len(y_preds)):
            result[i] = sign(y_preds[i])
        return result


    @property
    def x_fit(self):
        return self._x_train
    @property
    def y_fit(self):
        return self._y_train
    
    def _get_weight(self):
        # 【p139，公式8.3，8.4，8.5】
        weight_factors = self._weight*np.exp(-self._alpha*self._y_train*self._clf.predict(self._x_train))
        z = sum(weight_factors)
        new_wf = weight_factors / z
        return new_wf

x_train = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y_train = np.array([1, 1, 1,-1,-1,-1, 1, 1, 1,-1])
ada = Adaboost(10)
ada.fit(x_train, y_train)
print(ada.predict(x_train))
