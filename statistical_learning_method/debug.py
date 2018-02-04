import math

class Em(object):
    def __init__(self, epsilon0=0.001, epsilon1=0.001):
        self._epsilon0 = epsilon0
        self._epsilon1 = epsilon1
        self._max_iteration = 3
        self._nu = None
        self._pi = 0
        self._p = 0
        self._q = 0

        self._nus = []
        self._thetas = []
    @property
    def theta(self):
        return self._pi, self._p, self._q
    def train(self, y_train, pi, p, q):
        length = len(y_train)
        self._pi = pi
        self._p = p
        self._q = q
        self._thetas.append((pi, p, q))
        self._nu = [0]*length
        for i in range(self._max_iteration):
            # 【p156，公式9.5】
            for j in range(length):
                b = self._pi * math.pow(self._p, y_train[j]) * math.pow(1 - self._p, 1 - y_train[j])
                c = (1 - self._pi) * math.pow(self._q, y_train[j]) * math.pow(1 - self._q, 1 - y_train[j])
                self._nu[j] = b / (b + c)
            # 【p156，公式9.6】
            self._pi = 1. / length * sum(self._nu)
            # 【p156，公式9.7】
            self._p = sum([self._nu[k] * y_train[k] for k in range(length)]) / sum(self._nu)
            # 【p156，公式9.8】
            self._q = sum([(1 - self._nu[k]) * y_train[k] for k in range(length)]) \
                        / sum([(1 - self._nu[k]) for k in range(length)])
            self._thetas.append((self._pi, self._p, self._q))
            print((self._pi, self._p, self._q))
            if self.is_stop():
                print('满足停机条件，终止循环。%s of %s' % (i, self._max_iteration))
                break

    def is_stop(self):
        # 【p158，中部步骤（4）】停止条件，Q函数还没懂要怎么实现
        pi0, p0, q0 = self._thetas[-2]
        pi1, p1, q1 = self._thetas[-1]
        if all([pi1- pi0 < self._epsilon0, p1 - p0 < self._epsilon0, q1 - q0 < self._epsilon0]):
            return True
        return False
    def q_fun(self):
        # Q函数还没懂要怎么实现
        pass

x_train = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]
em = Em()
pi, p, q = (0.5, 0.5, 0.5)
print('设置初值：pi=%.2f，p=%.2f，q=%.2f' % (pi, p, q))
em.train(x_train, pi, p, q)
pi, p, q = (0.4, 0.6, 0.7)
print('设置初值：pi=%.2f，p=%.2f，q=%.2f' % (pi, p, q))
em.train(x_train, pi, p, q)
