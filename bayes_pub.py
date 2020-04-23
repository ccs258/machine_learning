# -*- coding: utf-8 -*-
# @Time : 2020/4/23 19:51
# @Author : ccs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter
import math

# data
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, :])
    # print(data)
    return data[:,:-1], data[:,-1]


class NaiveBayes:
    def __init__(self):
        self.model = None

    # 数学期望
    @staticmethod
    def mean(X):
        return sum(X) / float(len(X))

    # 标准差（方差）
    def stdev(self, X):
        avg = self.mean(X)
        return math.sqrt(sum([pow(x - avg, 2) for x in X]) / float(len(X)))

    # 概率密度函数
    def gaussian_probability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean, 2) /
                              (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    # 处理X_train
    def summarize(self, train_data):
        print("train_data is ",train_data)
        summaries = [(self.mean(i), self.stdev(i)) for i in zip(*train_data)]
        #train_data是训练集样本，summaries是针对训练样本的每列进行求均值方差，以进行高斯分布的模型建立；
        print("summaries is ",summaries)
        return summaries

    # 分类别求出数学期望和标准差
    def fit(self, X, y):
        labels = list(set(y))
        data = {label: [] for label in labels}
        for f, label in zip(X, y):
            data[label].append(f)  #构造字典{特征标签1：特征样本集合1，特征标签2：特征样本集合2}
        self.model = {
            label: self.summarize(value)
            for label, value in data.items()
        }
        print("self.model is ",self.model)
        return 'gaussianNB train done!'

    # 计算概率
    def calculate_probabilities(self, input_data):
        # summaries:{0.0: [(5.0, 0.37),(3.42, 0.40)], 1.0: [(5.8, 0.449),(2.7, 0.27)]}
        # input_data:[1.1, 2.2]
        probabilities = {}
        print("input_data is ",input_data)
        for label, value in self.model.items():
            print("value is ",value)
            probabilities[label] = 1
            for i in range(len(value)):
                mean, stdev = value[i]
                probabilities[label] *= self.gaussian_probability(
                    input_data[i], mean, stdev)
        return probabilities

    # 类别
    def predict(self, X_test):
        # {0.0: 2.9680340789325763e-27, 1.0: 3.5749783019849535e-26}
        print("self.calculate_probabilities(X_test) is ",self.calculate_probabilities(X_test))
        label = sorted(
            self.calculate_probabilities(X_test).items(),
            key=lambda x: x[-1])[-1][0]
        return label

    def score(self, X_test, y_test):
        right = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right += 1

        return right / float(len(X_test))

X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

####
print("X_test[0], y_test[0] is",X_test[0], y_test[0])
model = NaiveBayes()

model.fit(X_train, y_train)

print("model.predict is ",model.predict([4.4,  3.2,  1.3,  0.2]))

model.score(X_test, y_test)


"""
train_data is  [array([4.5, 2.3, 1.3, 0.3]), array([5.4, 3.4, 1.7, 0.2]), array([5.4, 3.7, 1.5, 0.2]), array([4.7, 3.2, 1.3, 0.2]), array([4.4, 3. , 1.3, 0.2]), array([4.8, 3.4, 1.9, 0.2]), array([5. , 3.5, 1.3, 0.3]), array([5. , 3.4, 1.6, 0.4]), array([4.4, 2.9, 1.4, 0.2]), array([5.4, 3.9, 1.7, 0.4]), array([5.1, 3.8, 1.9, 0.4]), array([5. , 3.3, 1.4, 0.2]), array([5. , 3.6, 1.4, 0.2]), array([4.6, 3.4, 1.4, 0.3]), array([5. , 3.4, 1.5, 0.2]), array([4.6, 3.2, 1.4, 0.2]), array([5.5, 4.2, 1.4, 0.2]), array([5. , 3.5, 1.6, 0.6]), array([5.4, 3.9, 1.3, 0.4]), array([4.8, 3. , 1.4, 0.1]), array([4.8, 3.1, 1.6, 0.2]), array([5.1, 3.8, 1.5, 0.3]), array([5.1, 3.3, 1.7, 0.5]), array([5.2, 4.1, 1.5, 0.1]), array([5. , 3.2, 1.2, 0.2]), array([5.1, 3.5, 1.4, 0.2]), array([5.5, 3.5, 1.3, 0.2]), array([4.8, 3.4, 1.6, 0.2]), array([4.9, 3.1, 1.5, 0.2]), array([5.7, 3.8, 1.7, 0.3]), array([4.6, 3.1, 1.5, 0.2]), array([4.7, 3.2, 1.6, 0.2]), array([4.6, 3.6, 1. , 0.2])]
summaries is  [(4.972727272727272, 0.33236774747142456), (3.415151515151514, 0.37101107728395566), (1.478787878787879, 0.18870784820533654), (0.2545454545454545, 0.10756508696544757)]




###
Ubuntu git代理修改后，git clone失败Could not resolve proxy: proxy.server.com 重置代理方法
原创悲恋花丶无心之人 最后发布于2018-07-05 13:14:47 阅读数 4554  收藏
展开

Pytorch使用教程和范例
从入门Pytorch到掌握Pytorch，只需跟着博主走！
悲恋花丶无心之人
¥9.90
订阅
当我们修改完git 代理时，git clone往往出现错误，此时如果想重置代理，卸载git是没有用的，而是重置git代理

因此，我们只需要执行以下两句命令即可：

git config --global --unset http.proxy
git config --global --unset https.proxy
然后我们就可以重新设置git代理了～
————————————————
版权声明：本文为CSDN博主「悲恋花丶无心之人」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_36556893/article/details/80925388
"""