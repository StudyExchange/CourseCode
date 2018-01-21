# 表4.1 训练数据
import numpy as np
import pandas as pd
from IPython.display import display

data = [
    [1, 'S', -1],
    [1, 'M', -1],
    [1, 'M', 1],
    [1, 'S', 1],
    [1, 'S', -1],
    [2, 'S', -1],
    [2, 'M', -1],
    [2, 'M', 1],
    [2, 'L', 1],
    [2, 'L', 1],
    [3, 'L', 1],
    [3, 'M', 1],
    [3, 'M', 1],
    [3, 'L', 1],
    [3, 'L', -1],
]

data_pd = pd.DataFrame(data, columns=['X1', 'X2', 'Y'])
# data_pd['X2'] = data_pd['X2'].map({'S': 0, 'M': 1, 'L': 2})
display(data_pd)

x_data = data_pd[['X1', 'X2']].as_matrix()
y_data = data_pd['Y'].as_matrix()

print(x_data.shape)
print(y_data.shape)

x_test = [2, 'S']
print('x_test:', x_test)


class NativeBayes(object):
    def __init__(self):
        self._py = {}
        self._pxy = {}
    @property
    def py(self):
        return self._py
    @property
    def pxy(self):
        return self._pxy
    
    def train(self, datas, labels):
        label_set = set(labels)
        # 【p50，算法4.1第（1）步】先验概率P(Y=Ck)
        py = {}
        for c in label_set:
            py[c] = (list(labels).count(c), len(labels))
        # 【p50，算法4.1第（1）步】条件概率P(X=Ajl | Y=Ck)
        pxy = {}
        for c in label_set:
            c_indexes = [i for i in range(len(labels)) if c == labels[i]] # 按label对数据分割，然后再求条件概率
            for i in range(datas[c_indexes].shape[1]):
                xi = datas[c_indexes, i]
                xi_set = set(xi)
                for j in xi_set:
                    pxy[(j, c)] = (list(xi).count(j), len(xi))
        self._py = py
        self._pxy = pxy
        self._label_set = label_set

    def predict(self, data):
        # 【p50，算法4.1第（2）步】计算各个py
        py = []
        for c in self._label_set:
            pxi = 1
            for xi in data:
                pxi = pxi * self._pxy[(xi, c)][0] / self._pxy[(xi, c)][1]
            py.append(self._py[c][0] / self._py[c][1] * pxi)
        print(py)
        # 【p50，算法4.1第（3）步】返回概率最大的label
        return list(self._label_set)[np.argsort(py)[-1]]

nb = NativeBayes()
print('先验概率P(Y=Ck)：')
nb.train(x_data, y_data)
for key in nb.py.keys():
    print(key, nb.py[key])

print('条件概率P(X=Ajl | Y=Ck)：')
for key in nb.pxy.keys():
    print(key, nb.pxy[key])

print('predict结果：%s' % nb.predict(x_test))
