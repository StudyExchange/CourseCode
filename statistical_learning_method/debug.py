import math

class MarkovDirectly(object):
    def probability(self, a, b, pi, output):
        an = len(a[0]) # a的状态的数量
        bn = len(b[0]) # b的状态的数量
        m = len(output) # 观测序列的长度
        inputs = []
        total_count = int(math.pow(an, m))
        result = 0
        for i in range(total_count): # i表示总的循环数
            # for j in range(an): # j表示pi的index
                # index_li = [j]
            index_li = MarkovDirectly.get_sequence_index(i, an, m)
            print('q_t的集合：%s' % index_li, end='。') # 即本次循环的状态集合，即本次循环的选中的盒子的集合
            prob = pi[index_li[0]] # pi(i1)
            for k in range(m - 1):
                prob *= b[index_li[k]][output[k]] # b_i1(o1), b_i2(o2)
                print('i%s->i%s' % (k, k+1), end='  ')
                prob *= a[index_li[k]][index_li[k+1]] # a_i1_i2, a_i2_i3
            prob *= b[index_li[m - 1]][output[m - 1]] # b_i3(o3)
            print('  %.5f' % prob)
            result += prob
        return result


    @staticmethod
    def get_sequence_index(num, an, m):
        if num:
            index_li = []
            while num:
                quotient, remainder = divmod(num, an)
                num = quotient
                index_li.append(remainder)
            if len(index_li) < m:
                index_li.extend([0]*(m - len(index_li)))
            return index_li
        else:
            return [0]*m

    def is_skip(self, indexes):
        pass


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

mf = MarkovDirectly()
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

mf = MarkovDirectly()
print(mf.probability(a, b, pi, output))

