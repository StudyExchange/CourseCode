import math

class MarkovForward(object):
    def probability(self, a, b, pi, output):
        an = len(a[0]) # a的状态的数量 = len(pi)
        bn = len(b[0]) # b的状态的数量
        m = len(output) # 观测序列的长度
        
        alphas = []
        alpha1 = [0] * an
        # alpha1
        for i in range(an):
            alpha1[i] = pi[i] * b[i][output[0]]
        print(alpha1)
        alphas.append(alpha1)
        # alpha2到alpha_T
        for i in range(1, m):
            alpha = [0] * an
            for j in range(an):
                alpha[j] = 0
                for k in range(an):
                    alpha[j] += alphas[-1][k] * a[k][j]
                print('%.3f x %.3f' % (alpha[j], b[j][output[i]]), end=' = ')
                alpha[j] *= b[j][output[i]]
                print('%.4f' % alpha[j])
            alphas.append(alpha)
        
        result = sum(alphas[-1])
        return result


# 测试数据1：测试数据：p177，例10.2（主要的测试数据）
a = [
    [0.5, 0.2, 0.3],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 0.5]
]
b = [
    [0.5, 0.5],
    [0.4, 0.6],
    [0.7, 0.3]
]
pi = [0.2, 0.4, 0.4]
output = [0, 1, 0]

mf = MarkovForward()
print(mf.probability(a, b, pi, output))


# 测试数据2：测试数据：p173，例10.1（盒子和球模型）（用来与前向和后向算法做对照可以相互印证，结果是否正确）
a = [
    [0, 1, 0, 0],
    [0.4, 0, 0.6, 0],
    [0, 0.4, 0, 0.6],
    [0, 0, 0.5, 0.5]
]
b = [
    [0.5, 0.5],
    [0.3, 0.7],
    [0.6, 0.4],
    [0.8, 0.2]
]
pi = (0.25, 0.25, 0.25, 0.25)
output = [0, 0, 1, 1, 0]

mf = MarkovForward()
print(mf.probability(a, b, pi, output))

