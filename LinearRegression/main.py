# -*- coding: utf-8 -*-
# @Time    : 2022/9/6 10:33 AM
# @Author  : Hush
# @Email   : crush@tju.edu.cn
import numpy as np
import LinearRegression as LR
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


def data_process():
    # 加载波斯顿房价数据集
    data = pd.read_csv('data/housing.csv')
    column_header = list(data.columns.values)[0:-1]
    print(column_header)
    # print(data.isnull().sum())

    X, y = data.iloc[1:, :-1].values, data.iloc[1:, -1].values  # ## 获取数据的从1到最后一行，从0到倒数第2列
    # print(X.describe(), y.describe())
    # for i in range(len(column_header)):
    #     plt.scatter(X[:,i],y,s=20)
    #     plt.title(column_header[i])
    #     plt.show()
    ## 选择RM 和 LSTAT
    X = X[:, [5, -1]]
    # print(X.shape)
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
    epochs = 500
    ## 使用归一化效果
    train_X = AllNorm_3(x_train)
    test_X = AllNorm_3(x_test)

    LinearRegression = LR.LinearRegression(train_X, test_X)
    loss_record, acc_record, acc_test_record = LinearRegression.fit(LinearRegression.train_X, y_train, LinearRegression.test_X, y_test, epochs)
    # print(acc_record)
