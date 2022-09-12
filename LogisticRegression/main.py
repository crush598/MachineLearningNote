# -*- coding: utf-8 -*-
# @Time    : 2022/9/12 3:22 PM
# @Author  : Hush
# @Email   : crush@tju.edu.cn

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import LogisticRegression as LR


def data_process():
    # 加载乳腺癌数据集
    cancers = datasets.load_breast_cancer()
    feature_names = cancers.feature_names
    X = cancers.data
    y = cancers.target
    # print(X.shape, y.shape)
    # for i in range(len(feature_names)):
    #     plt.scatter(X[:,i],y,s=20)
    #     plt.title(feature_names[i]+ f"{i}")
    #     plt.show()
    X = X[:, [0,2,3,7,8,20,21,22,23,27]]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
    return x_train, x_test, y_train, y_test

def AllNorm_3(X):  # 基于均值和标准差
    m = X.shape[0]

    mean_vals = np.mean(X)
    std_vals = np.std(X)
    normDataSet = np.zeros(np.shape(X))
    # print(normDataSet.shape)

    normDataSet = X - np.tile(mean_vals, (m, 1))  # 在行方向重复minVals m次和列方向上重复minVals 1次
    normDataSet = normDataSet / np.tile(std_vals, (m, 1))
    return normDataSet


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = data_process()
    epochs = 1000

    train_X = AllNorm_3(x_train)
    test_X = AllNorm_3(x_test)

    LinearRegression = LR.LogisticRegression(train_X, test_X)
    loss_record, acc_record, acc_test_record = LinearRegression.fit(LinearRegression.train_X, y_train,
                                                                    LinearRegression.test_X, y_test, epochs)
    print(acc_test_record[-1])