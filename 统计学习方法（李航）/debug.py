import numpy as np

def classify(i, y, gram, alfa, b):
    # 【p34，算法2.2 第(3)步】分类函数
    result = y[i]*(np.sum(alfa*y*gram[i, :]) + b)
    return result

def loss(x, y, w, b):
    # 【p27，公式2.4】损失函数L(w, b)
    distances = np.squeeze(np.dot(np.transpose(w), x) + b)
    distances_y = np.transpose(y)*distances
    losses = 0
    for d in distances_y:
        if d < 0: # 损失函数是所有误分类样本的损失的总和
            losses += d
    result = - losses
    return result

def data_generator(x, y):
    # 用于持续生成数据
    index = 0
    while True:
        # x[:, index]切片的结果是一个行向量，需要reshape成列向量
        xi = x[:, index].reshape(x.shape[0], 1)
        yield index, xi, y[index] # 与算法2.1相比，这里多返回一个index，以便知道返回的是第几个样本
        index = index + 1
        index = index % len(y)

def gram_matrix(x):
    # 【p34，例2.2 第(2)步】gram矩阵
    m = x.shape[1] # m表示样本数
    g = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            g[i, j] = np.squeeze(np.dot(x[:, i], x[:, j]))
    return g

def weight(alfa, x, y):
    # 【p33，公式2.14】
    result = np.sum(alfa*y*x, axis=1)
    return result

# 初始化
x = np.array([[3, 4, 1],
              [3, 3, 1]])
y = np.array([1, 1, -1])

alfa = np.zeros((len(y),))
w = np.zeros((2, 1))
b = 0
eta = 1


# 保存每个迭代损失函数的损失
losses = []
# 数据生成器
gen = data_generator(x, y)
gram = gram_matrix(x)
print('gram矩阵：\n', gram)


count = 0 # 累计连续正确分类样本的次数
while True:
    i, xi, yi = next(gen)
    # 【p34，算法2.2第（3）步】分类
    result = classify(i, y, gram, alfa, b)
    if result <= 0:
        # 【p34，算法2.2第（3）步】更新alfa，b
        alfa[i] = alfa[i] + eta
        b = b + eta*yi
        print(alfa, b)
        # 额外计算一下w和损失函数
        w = weight(alfa, x, y)
        losses.append(loss(x, y, w, b))
        count = 0 # 如果有无分类的样本，计数清零
    else:
        count += 1
    i += 1
    if count == len(y): # 算法第4步，当这个次数等于样本总数时，说明全部的样本点都被正确分类了
        break

print('最后结果：')
print('loss：', losses)
print('w：', w)
print('b：', b)