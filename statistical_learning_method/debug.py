import numpy as np
import random

class Smo(object):
    def __init__(self, c=10):
        # self._w = None
        self._b = 0
        self._c = c
        self._epsilon = 0.001
        # self._xi = 0.001
        self._max_interaction = 100

    def train(self, x, y):
        self._init_parameters(x, y)
        for i in range(self._max_interaction):
            # 【参照@WenDesi的实现】把0<alpha<C的alpha放在list的前面，剩余的放在后面。
            # 这样实现p129第1段中描述。
            indexes = list(range(self._m))
            condidate_indexes = list(filter(lambda i: self._alpha[i] > 0 and self._alpha[i] < self._c, indexes))
            remaining_indexes = list(set(indexes) - set(condidate_indexes))
            entire_list = condidate_indexes + remaining_indexes
            for j in entire_list:
                if self._is_stop():
                    print('Finished at time: %s' % i)
                    return
                i1 = j
                i2 = self._choose_second_parameter(i1)
                print(i1, i2)
                # 【p126，最后一段，求L和H的公式】
                if self._y[i1] != self._y[i2]:
                    low = max(0, self._alpha[i2] - self._alpha[i1])
                    high = min(self._c, self._c + self._alpha[i2] - self._alpha[i1])
                else:
                    low = max(0, self._alpha[i2] + self._alpha[i1] - self._c)
                    high = min(self._c, self._alpha[i2] + self._alpha[i1])
                if low == high:
                    continue
                # 【p127，公式7.107】eta = K11 + K12 - 2K12
                eta = self._k(self._x[i1], self._x[i1]) + self._k(self._x[i2], self._x[i2]) - 2 * self._k(self._x[i1], self._x[i2])
                # 【p127，公式7.106】未经剪辑的alpha2_new
                alpha2_new = self._alpha[i2] + self._y[i2] * (self._error(i1) - self._error(i2)) / eta
                # 剪辑alpha2_new
                # alpha2_new = self._clip_alpha2_new(alpha2_new, low, high)
                # 【p127，公式7.108】
                alpha2_new = min(high, alpha2_new)
                alpha2_new = max(low, alpha2_new)
                # 【p127，公式7.109】有alpha2_new求alpha1_new
                alpha1_new = self._alpha[i1] + self._y[i1] * self._y[i2] * (self._alpha[i2] - alpha2_new)
                # 【p130，公式7.115】计算b2_new
                b1_new = -self._error(i1) - self._y[i1] * self._k(self._x[i1], self._x[i1]) * (alpha1_new - self._alpha[i1]) \
                        -self._y[i2] * self._k(self._x[i2], self._x[i1]) * (alpha2_new - self._alpha[i2]) + self._b
                # 【p130，公式7.116】计算b1_new
                b2_new = -self._error(i2) - self._y[i1] * self._k(self._x[i1], self._x[i2]) * (alpha1_new - self._alpha[i1]) \
                        -self._y[i2] * self._k(self._x[i2], self._x[i2]) * (alpha2_new - self._alpha[i2]) + self._b
                # 【p130，中部，关于b_new的取值方法】
                if alpha1_new > 0 and alpha1_new < self._c:
                    b_new = b1_new
                elif alpha2_new > 0 and alpha2_new < self._c:
                    b_new = b2_new
                else:
                    b_new = (b1_new + b2_new) / 2
                self._b = b_new
                # 更新alpha1和alpha2
                self._alpha[i1] = alpha1_new
                self._alpha[i2] = alpha2_new
                print(self._alpha)
                print(self._weight(), self._b)
        print('max interaction')

    def predict(self, x):
        results = [0] * len(x)
        for i, sample in enumerate(x):
            # 【p124，公式7.94】分类决策函数
            y_hat = self._b
            for j in range(self._m):
                temp = self._alpha[j] * self._y[j] * self._k(sample, self._x[j])
                y_hat += temp
            results[i] = self._sign(y_hat)
        return results

    def _weight(self):
        # 【p33，公式2.14】
        result = np.dot(np.array(self._alpha) * np.array(self._y), np.array(self._x))
        return result

    def _sign(self, x):
        if x >= 0:
            return 1
        else:
            return -1

    def _init_parameters(self, x, y):
        self._x = x
        self._y = y

        self._m = len(self._x) # m 表示样本的数量
        self._n = len(self._x[0]) # n 表示特征的数量
        self._alpha = [0] * self._m

    def _clip_alpha2_new(self, alpha2_new, low, high):
        if high < low:
            raise 'high不能小于low，程序有错，high：%.2f，low：%.2f' % (high, low)
        # 【p127，公式7.108】
        if alpha2_new > high:
            return high
        elif alpha2_new >= low and alpha2_new <= high:
            return alpha2_new
        else:
            return low

    def _gx(self, index):
        # 【p127，公式7.104】gx
        gxi = self._b
        for j in range(self._m): # 为了与书中对应，使用j作为迭代变量
            gxi += self._alpha[j] * self._y[j] * self._k(self._x[j], self._x[index])
        return gxi

    def _k(self, xi, xj):
        """线性核
        """
        return float(np.dot(xi, xj))

    def _error(self, index):
        # 【p127，公式7.105】单个样本的误差
        return self._gx(index) - self._y[index]

    def _is_satisfy_kkt(self, index):
        """
        检查是否满足KKT条件。按照论文第8页最后一段的说法，epsilon一般取0.001，会加速收敛的过程。
        """
        ygx = self._y[index] * self._gx(index)
        # 【p128，公式7.111】-epsilon <= alpha <= epsilon
        if self._alpha[index] >= -self._epsilon and self._alpha[index] <= self._epsilon:
            return ygx >= 1
        # 【p129，公式7.112】epsilon < alpha < C-epsilon
        elif self._alpha[index] > self._epsilon and self._alpha[index] < self._c - self._epsilon: 
            return ygx == 1
        # 【p129，公式7.113】C-epsilon <= alpha <= C+epsilon
        elif self._alpha[index] >= self._c - self._epsilon and self._alpha[index] <= self._c + self._epsilon:
            return ygx <= 1
        else:
            raise 'alpha取到了异常值:%.3f，alpha的取值范围应该是：%.3f<=alpha<=%.3f。' \
                % (self._alpha[index], -self._epsilon, self._c + self._epsilon)

    def _is_stop(self):
        """
        检查是否所有的样本都满足KKT条件。如果，都满足，则返回True，即停机并返回最终的结果。否则，返回False。
        【p130，算法7.5，第（3）步】停机条件
        """
        for i in range(self._m):
            if not self._is_satisfy_kkt(i):
                return False
        return True

    def _choose_second_parameter(self, first_parameter_index):
        """
        选择第2个变量。【p129，2. 第2个变量的选择】
        """
        # # 先计算全部样本的error
        # errors = [self._error(i) for i in range(self._m)]
        # # 再计算E1与每个样本的E
        # error_diffs = [abs(errors[first_parameter_index] - errors[i]) for i in range(self._m)]
        # # 排序，并选择最大的error_diff
        # sorted_error_diff_indexes = np.argsort(error_diffs)
        # # 如果error_diffs最大的那个变量是第1个变量本身，则选择第2大的那个变量
        # if sorted_error_diff_indexes[-1] == first_parameter_index:
        #     return int(sorted_error_diff_indexes[-2])
        # else:
        #     return int(sorted_error_diff_indexes[-1])
        entire = list(range(self._m))
        entire.remove(first_parameter_index)
        i = int(random.random() * len(entire))
        return entire[i]
        # return min(first_parameter_index + 1, self._m)

    def _choose_parameters(self):
        indexes = list(range(self._m))
        # 先从0<alpha<C对应的样本中选择第1个不符合KKT条件的样本，并作为第1个变量。
        # 1. 【p128，最后一段说，“选取违反KKT条件最严重的样本点”，因为没有看懂如何判断最严重，
        # 所以，本程序中，直接选择第1个违反KKT条件的样本。】
        # 2. 【SMO论文p10、p11中，说随机选择一个违反KKT条件样本，本程序并没有按这种方式实现。】
        condidate_indexes = list(filter(lambda i: self._alpha[i] > 0 and self._alpha[i] < self._c, indexes))
        for i in condidate_indexes:
            if self._is_satisfy_kkt(i):
                continue
            p2 = self._choose_second_parameter(i)
            return i, p2

        # 如果搜索的0<alpha<C对应的样本都满足KKT条件，那么从剩余的样本中选择第1个不符合KKT条件的样本，
        # 并作为第1个变量。
        remaining_indexes = list(set(indexes) - set(condidate_indexes))
        for i in remaining_indexes:
            if self._is_satisfy_kkt(i):
                continue
            i2 = self._choose_second_parameter(i)
            return i, i2

x_data = [
    [14, 13],
    [4, 3],
    [3, 3],
    [2.5, 2.5],
    [2, 2],
    [1.5, 1.5],
    [1, 1]
]
y_data = [
    1,
    1,
    1,
    -1,
    -1, 
    -1,
    -1]
x_val = [
    [0, 0],
    [0, 2],
    [2, 0],

    [0, 4],
    [4, 0],

    [0, 6],
    [6, 0]
]
y_val = [-1, -1, -1, 1, 1, 1, 1]

smo = Smo()
try:
    smo.train(x_data, y_data)
    y_pre = smo.predict(x_val)
except Exception as ex:
    print(ex)
print(y_pre)
print(y_val)
