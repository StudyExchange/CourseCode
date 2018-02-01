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


import math
# 对于指定数据集（只根据label的数据），求基尼指数
def gini(labels):
    # 【p69，公式5.24】给定样本集合D，求基尼系数
    lenght = len(labels)
    classes = list(set(labels))
    ckd_squre = []
    for c in classes:
        ckd_squre.append(math.pow(list(labels).count(c)/lenght, 2))
    result = 1 - sum(ckd_squre)
    return result

assert gini([0,]) == 0
assert gini([0, 0]) == 0
assert gini([0, 1]) == 0.5
assert gini([0, 0, 0, 1]) == 0.375


# 对于某个特征条件下，求集合的基尼指数
def gini_feature(features, labels, class_name):
    # 【p69，公式5.24】给定样本集合D，求基尼系数
    lenght = len(labels)
    # 使用特征区分数据
    label_index0 = [labels[i] for i in range(lenght) if features[i] == class_name]
    label_index1 = [labels[i] for i in range(lenght) if features[i] != class_name]
    # 【p70，公式5.25】对于某个特征条件下，求集合的基尼指数
    # print((gini(label_index0)), (gini(label_index1)))
    result = len(label_index0)/lenght*(gini(label_index0)) + len(label_index1)/lenght*(gini(label_index1))
    return result

test_f = [10, 10, 10, 10, 11, 11, 11, 11]
test_l = [1,  1,  1,  1,  1,  1,  1,  1]
print(gini_feature(test_f, test_l, 10))
assert gini_feature(test_f, test_l, 10) == 0

test_f = [10, 10, 10, 10, 11, 11, 11, 11]
test_l = [1,  1,  0,  0,  1,  1,  0,  0]
print(gini_feature(test_f, test_l, 10))
assert gini_feature(test_f, test_l, 10) == 0.5

test_f = [10, 10, 10, 10, 11, 11, 11, 11]
test_l = [1,  1,  1,  0,  0,  1,  1,  1]
print(gini_feature(test_f, test_l, 10))
assert gini_feature(test_f, test_l, 10) == 0.375


class Node(object):
    def __init__(self, features, labels):
        self._features = features # 叶子节点的数据的x_data
        self._labels = labels # 叶子节点的数据的y_data
        self._label = None # 叶子节点的类
        self._lchild = None
        self._rchild = None
        self._feature_index = None
        self._split_value = None
        self._fixed_indexes = []
        if len(self._features) > 0:
            self._feature_indexes = list(range(len(self._features[0])))
    @property
    def features(self):
        return self._features
    @features.setter
    def features(self, value):
        self._features = value
    @property
    def labels(self):
        return self._labels
    @labels.setter
    def labels(self, value):
        self._labels = value
    @property
    def label(self):
        return self._label
    @label.setter
    def label(self, value):
        self._label = value

    @property
    def lchild(self):
        return self._lchild
    @lchild.setter
    def lchild(self, value):
        self._lchild = value
    @property
    def rchild(self):
        return self._rchild
    @rchild.setter
    def rchild(self, value):
        self._rchild = value
        
    @property
    def feature_index(self):
        return self._feature_index
    @feature_index.setter
    def feature_index(self, value):
        self._feature_index = value
    @property
    def split_value(self):
        return self._split_value
    @split_value.setter
    def split_value(self, value):
        self._split_value = value
    @property
    def fixed_indexes(self):
        return self._fixed_indexes
    @fixed_indexes.setter
    def fixed_indexes(self, value):
        self._fixed_indexes = value

    @property
    def feature_indexes(self):
        return self._feature_indexes
    def split(self, indexes):
        features = self._features[indexes, :]
        labels = self._labels[indexes]
        return features, labels
    def is_leaf(self):
        if self._label: # label有值，说明是叶子节点
            return True
        return False
    def printt(self):
        if self.is_leaf():
            print(' '*4*len(self._fixed_indexes), end='') # 每级4个空格
            print((self._label))
        else:
            print(' '*4*len(self._fixed_indexes), end='') # 每级4个空格
            print((self._feature_index, self._split_value))


def cart(node, min_sample_count=2, min_gini=0.1):
    # 【p71，第一段，停止条件：样本数小于预定阈值（默认值2，即最少一个样本）】
    if len(node.labels) < min_sample_count:
        # 参考【p63，算法5.2第（2）步】获取实例数最大的类
        node.label = max(set([1,2,4]))
        return node
    # 【p71，第一段，停止条件：样本属于同一个类】
    if len(list(set(node.labels))) == 1:
        node.label = node.labels[0]
        return node
    # 【p71，第一段，停止条件：小于基尼指数阈值】
    if gini(node.labels) < min_gini:
        # 书中没有说明，这里我取实例数最大的类
        node.label = max(set([1,2,4]))
        return node
    # 开始选择切分点(feature_index, split_value)
    split_points = []
    # 获取没有固定的特征，即可以用于特征选择的特征
    unfixed_indexes = [i for i in node.feature_indexes if i not in node.fixed_indexes]
    for i in unfixed_indexes:
        single_feature = node.features[:, i]
        classes = list(set(single_feature))
        for j in classes:
            gini_index = gini_feature(single_feature, node.labels, j)
            split_points.append((i, j, gini_index))
    # print(split_points) # 这里检查得到的基尼指数与【p71，例5.4中的数据是否一致】
    # 获取最小的基尼指数
    split_points = sorted(split_points, key=lambda g: g[2])
    print(split_points[0])
    node.feature_index = split_points[0][0]
    node.split_value = split_points[0][1]
    # 区分特征，递归生成节点
    child_fixed_indexes = node.fixed_indexes.copy() + [node.feature_index]
    this_feature = node.features[:, node.feature_index]
    l_indexes = [i for i in range(len(this_feature)) if this_feature[i] == node.split_value] # 左节点存的是split_value的值
    lf, ll = node.split(l_indexes)
    l_child = Node(lf, ll)
    l_child.fixed_indexes = child_fixed_indexes
    node.lchild = cart(l_child, min_sample_count=2, min_gini=0.1)
    
    r_indexes = [i for i in range(len(this_feature)) if this_feature[i] != node.split_value]
    rf, rl = node.split(r_indexes)
    r_child = Node(rf, rl)
    r_child.fixed_indexes = child_fixed_indexes
    node.rchild = cart(r_child, min_sample_count=2, min_gini=0.1)
    return node
    
root_node = Node(x_data, y_data)
test_tree = cart(root_node)
pass


# 中序遍历，打印树
def print_tree(node):
    node.printt()
    if not node.is_leaf():
        print_tree(node.lchild) # 左节点存的是split_value的值
        print_tree(node.rchild)
print_tree(test_tree)
