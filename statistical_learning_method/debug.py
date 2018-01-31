import pdb

import numpy as np
import pandas as pd
from IPython.display import display

# 表5.1 贷款申请样本数据表
loan_application = [
    (1, '青年', '否', '否', '一般', '否'),
    (2, '青年', '否', '否', '好', '否'),
    (3, '青年', '是', '否', '好', '是'),
    (4, '青年', '是', '是', '一般', '是'),
    (5, '青年', '否', '否', '一般', '否'),
    (6, '中年', '否', '否', '一般', '否'),
    (7, '中年', '否', '否', '好', '否'),
    (8, '中年', '是', '是', '好', '是'),
    (9, '中年', '否', '是', '非常好', '是'),
    (10, '中年', '否', '是', '非常好', '是'),
    (11, '老年', '否', '是', '非常好', '是'),
    (12, '老年', '否', '是', '好', '是'),
    (13, '老年', '是', '否', '好', '是'),
    (14, '老年', '是', '否', '非常好', '是'),
    (15, '老年', '否', '否', '一般', '否')
]

df = pd.DataFrame(loan_application, columns=['ID', '年龄', '有工作', '有自己的房子', '借贷情况', '类别'])
# display(df)

# 将类型转换为int类型，使用100、101、10、11、12等数字，而不是0、1、2等是为了便于在计算过程中区分各个类别
df['类别'] = df['类别'].map({'否': 100, '是': 101}).astype(int)
df['年龄'] = df['年龄'].map({'青年': 10, '中年': 11, '老年': 12}).astype(int)
df['有工作'] = df['有工作'].map({'否': 20, '是': 21}).astype(int)
df['有自己的房子'] = df['有自己的房子'].map({'否': 30, '是': 31}).astype(int)
df['借贷情况'] = df['借贷情况'].map({'一般': 40, '好': 41, '非常好': 42})
# display(df)

x_data = df[['年龄', '有工作', '有自己的房子', '借贷情况']].as_matrix()
y_data = df['类别'].as_matrix()

print(x_data)
print(y_data)
print(x_data.shape)
print(y_data.shape)

# （1）经验熵H(D)
import math

def empirical_entropy(labels):
    # labels：int类型的list
    lenght = len(labels)
    class_names = set(labels)
    # 用字典来统计各个值出现的次数
    class_group = {}
    # 初始化字典，存储各个类别的数量都为0
    for c in class_names:
        class_group[c] = list(labels).count(c)
    # 汇总各个类别，【p62，公式5.7】
#     entropies = []
#     for g in class_group.values():
#         entropies.append(- g / lenght * math.log2(g / lenght))
#     entropy = sum(entropies)
    entropy = sum([- g / lenght * math.log2(g / lenght) for g in class_group.values()]) #这一行代码等同于前面4行
    return entropy

entropy = empirical_entropy(y_data)
print('（1）经验熵H(D) = %.3f' % entropy)

# （2）条件经验熵H(D|A)

def conditional_empirical_entropy(feature, labels):
    # feature：单个特征的int类型的list，int是map之后的类别，支持多个特征取条件经验熵
    # labels：int类型的list，int是map之后的类别
    lenght = len(labels)
    classes = set(feature)
    # 按feature对数据进行分组
    feature_group = {}
    for c in classes:
        indexes = [i for i in range(len(feature)) if feature[i] == c]
        feature_group[c] = indexes
#     print(feature_group)
    entropies = []
    # 【p62，公式5.8】
    for indexes in feature_group.values():
        di_d = len(indexes) / lenght
        entropies.append(di_d*empirical_entropy(labels[indexes]))
    entropy = sum(entropies)
    return entropy

entropy = conditional_empirical_entropy(x_data[:, 0], y_data)
print('（2）条件经验熵H(D|A) = %.3f' % entropy)

# （3）信息增益g(D,A)
def information_gain(feature, labels):
    # 【p62，公式5.9】
    return empirical_entropy(labels) - conditional_empirical_entropy(feature, labels)

gain = information_gain(x_data[:, 0], y_data)
print('（3）信息增益g(D,A) = %.3f' % gain)

# 计算各个特征的信息增益
for i in range(x_data.shape[1]):
    print(information_gain(x_data[:, i], y_data))

def get_max_information_gain(features, labels):
    gains = [information_gain(features[:, i], labels) for i in range(features.shape[1])]
    sorted_indexes = np.argsort(gains)
#     print(sorted_indexes)
    return sorted_indexes[-1], gains[sorted_indexes[-1]]

max_feature_index, max_feature_gain = get_max_information_gain(x_data, y_data)
print('信息增益最大的index是：%s，对应特征是：A(%s)，信息增益是：%.3f' % (max_feature_index, max_feature_index + 1, max_feature_gain))

# （3）信息增益比gr(D,A)
def information_gain_ratio(feature, labels):
    # 【p62，公式5.9】
    return information_gain(feature, labels) / empirical_entropy(labels)

gain = information_gain_ratio(x_data[:, 0], y_data)
print('（3）信息增益比gr(D,A) = %.3f' % gain)

# 计算各个特征的信息增益比
for i in range(x_data.shape[1]):
    print(information_gain_ratio(x_data[:, i], y_data))

def get_max_information_gain_ratio(features, labels):
    gains = [information_gain_ratio(features[:, i], labels) for i in range(features.shape[1])]
    sorted_indexes = np.argsort(gains)
#     print(sorted_indexes)
    return sorted_indexes[-1], gains[sorted_indexes[-1]]

max_feature_index, max_feature_gain = get_max_information_gain_ratio(x_data, y_data)
print('信息增益最大的index是：%s，对应特征是：A(%s)，信息增益是：%.3f' % (max_feature_index, max_feature_index + 1, max_feature_gain))

class Node(object):
    def __init__(self, sub_space, labels):
        # 如果不是叶子节点，则feature_index=index(特征再数据集中横向的索引位置), pre_label=None
        # 如果  是叶子节点，则feature_index=None, pre_label=class
        self._sub_space = sub_space
        self._labels = labels
        self._feature_index = None
        self._pre_label = None
        self._father_node = None
        self._can_pruned = None # 用于保存修剪状态，可修剪：0，不可修剪：1
        self._children = {}
    # 属性
    @property
    def sub_space(self):
        return self._sub_space
    @property
    def labels(self):
        return self._labels

    @property
    def feature_index(self):
        return self._feature_index
    @feature_index.setter
    def feature_index(self, value):
        self._feature_index = value
    @property
    def pre_label(self):
        return self._pre_label
    @pre_label.setter
    def pre_label(self, value):
        self._pre_label = value
    @property
    def father_node(self):
        return self._father_node
    @father_node.setter
    def father_node(self, value):
        self._father_node = value
    @property
    def can_pruned(self):
        return self._can_pruned
    @can_pruned.setter
    def can_pruned(self, value):
        self._can_pruned = value

    @property
    def children(self):
        return self._children
    @children.setter
    def children(self, value):
        self._children = value
    
    # 方法
    def is_leaf(self):
        if self._feature_index is None and self._pre_label is not None:
            return True
        elif self._feature_index is not None and self._pre_label is None:
            return False
        else:
            raise Exception('节点状态异常')


def get_max_class(labels):
    classes = set(labels)
    class_group = []
    for c in classes:
        class_group.append(list(labels).count(c))
    max_feature_index = np.argsort(class_group)[-1]
    return list(classes)[max_feature_index]

test_labels = [2, 3, 3, 2, 1, 1, 1, 1]
class_name = get_max_class(test_labels)
assert 1 == class_name

def c45(features, labels, epsilon=0.1):
    assert len(features) > 0
    assert len(features) == len(labels)
    length = len(labels)
    classes = set(labels)
    # 【p65，算法5.3第（1）步】
    if len(classes) == 1:
        return Node(features, labels, None, classes.pop())
    # 【p65，算法5.3第（2）步】获取实例数最大的类
    if len(features[0]) == 0:
        max_class = get_max_class(labels)
        return Node(features, labels, None, max_class)
    # 【p65，算法5.3第（3）步】选择信息增益比最大的特征
    max_feature_index, max_feature_gain = get_max_information_gain_ratio(features, labels)
    # 【p65，算法5.3第（4）步】信息增益小于阈值
    if max_feature_gain < epsilon:
        max_class = get_max_class(labels)
        return Node(features, labels, None, max_class)
    # 【p65，算法5.3第（5）（6）步】构建多叉树的节点
    feature_indexes = list(range(features.shape[1]))
    feature_indexes.remove(max_feature_index) # 第（6）步，从数据集中剔除当前特征
    minus_features = features[:,feature_indexes]
    max_feature = features[:, max_feature_index] # 按增益最大的特征分割数据集
    max_classes = set(max_feature)
    # 按类型对数据集进行分割
    node = Node(features, labels, max_feature_index)
    for c in max_classes:
        indexes = [i for i in range(len(max_feature)) if max_feature[i] == c]
        print('feature:%s' % c)
        print(minus_features[indexes])
        print(labels[indexes])
        child_node = c45(minus_features[indexes], labels[indexes])
        # pdb.set_trace()
        node.children[c] = child_node
    return node


# 测试函数c45，第1个if分支
test_features = np.array([[], [], []])
test_labels = np.array([0, 0, 0])
test_tree = c45(test_features, test_labels)
assert isinstance(test_tree, Node)
assert test_tree.is_leaf()

# 测试函数c45，第2个if分支
test_features = np.array([[], [], []])
test_labels = np.array([0, 1, 1])
test_tree = c45(test_features, test_labels)
assert isinstance(test_tree, Node)
assert test_tree.is_leaf()
assert 101, test_tree.pre_label

# 测试函数c45，第3个if分支，使用前面整理好的数据：x_data和y_data
test_tree = c45(x_data, y_data, 0.45) # epsilon取0.45，让特征A(3)的增益0.42小于这个阈值
assert isinstance(test_tree, Node)
assert test_tree.is_leaf()
assert 101 == test_tree.pre_label

# 测试函数c45，剩余分支，使用前面整理好的数据：x_data和y_data
test_tree = c45(x_data, y_data, epsilon=0.1) # epsilon取默认值0.1
assert isinstance(test_tree, Node)
assert False == test_tree.is_leaf()

print('*'*8)
print(test_tree.sub_space)
print(test_tree.labels)




import queue
def get_leaves(node):
    # 层遍历树（用队列实现层遍历），获取所有的叶子节点
    que = queue.Queue()
    leaves = []
    que.put(node)
    while not que.empty():
        node = que.get()
        if node.is_leaf():
            leaves.append(node)
        else:
            for key in node.children.keys():
                que.put(node.children[key])
    return leaves

def set_leaves_all_soft(leaves):
    # 设置所有叶子节点的状态为待处理：0
    # 不可修剪状态为：1
    new_leaves = []
    for node in enumerate(leaves):
        new_leaves.append((node, 0))
    return new_leaves

def is_stop(leaves):
    # 如果全部得叶子节点全都不能再修剪，则停止修剪算法
    for node in enumerate(leaves):
        # 如果有一个节点可以修剪，则直接返回False
        if node[0] == 0:
            return False
    else:
        return True

leaves = get_leaves(test_tree)
assert len(leaves) == 3


def tree_loss(root_node, alpha):
    pass

def pruning_node():
    pass

def pruning(node):
    max_interaction = 100
    for i in range(max_interaction):
        leaves = get_leaves(node)
        leaves = set_leaves_all_soft(leaves)
        if is_stop(leaves):
            return node
        for leaf in leaves:
            pass
            # 当前树的损失函数
            # leaf这个节点的父节点剪枝之后的损失函数
            # 比较损失函数：
            # 之前大于之后：剪枝
            # 否则，设置这个父节点对应的这几个叶子节点的状态都是1

pruning(test_tree)
