# -*- coding: utf-8 -*-
# @Time    : 2022/9/12 3:21 PM
# @Author  : Hush
# @Email   : crush@tju.edu.cn

import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:

    def __init__(self, data, data_test):
        self.m = data.shape[0]
        self.cols = data.shape[1] + 1
        self.w = np.zeros(self.cols)
        self.b = np.ones(self.m).reshape(self.m, 1)
        self.lr = 0.001
        self.train_X = np.hstack([self.b, data])

        self.m_test = data_test.shape[0]
        self.b_test = np.ones(self.m_test).reshape(self.m_test, 1)
        self.test_X = np.hstack([self.b_test, data_test])

    def sigmoid(self, x):
        """
        逻辑函数
        """
        res = 1 / (1.0 + np.exp(-x))
        return np.clip(res, 1e-8, (1 - 1e-8))

    def predict(self, y):
        """
        使用模型进行预测标记值
        """
        y_pred = np.where(y > 0.5, 1, 0)
        return y_pred

    def fit(self, X, Y, X_test, Y_test, epochs):
        i = 0
        loss_record = []
        acc_record = []
        acc_test_record = []
        for i in range(epochs):
            ## predict
            y_hat = self.sigmoid(np.dot(X, self.w))

            ## compute loss
            loss = -(np.sum(Y * np.log(y_hat) + (1 - Y) * np.log(1 - y_hat))) / self.m
            loss_record.append(loss)

            ## compute gradiant
            grad = (np.dot(X.T, (y_hat - Y)))

            ## update weight----gradient_descent
            self.w = self.w - self.lr * grad

            acc = 1 - (np.sum(np.abs(self.predict(y_hat) - Y)) / self.m)
            acc_record.append(acc)

            ## test each epoch
            acc_test = self.test(X_test, Y_test)
            acc_test_record.append(acc_test)
        self.plot_history(loss_record, acc_record, acc_test_record, epochs)
        return loss_record, acc_record, acc_test_record

    def test(self, X, Y):
        acc_test = 1 - (np.sum(np.abs(self.predict(self.sigmoid(np.dot(X, self.w))) - Y)) / self.m_test)
        return acc_test

    def plot_history(self, loss_record, acc_record, acc_test_record, epochs):
        plt.figure(figsize=(10, 5))
        plt.subplot(2, 2, 1)
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.plot(range(epochs), loss_record, label='loss')
        plt.legend()
        plt.subplot(2, 2, 2)
        plt.xlabel('Epoch')
        plt.ylabel('acc_train')
        plt.plot(range(epochs), acc_record, label='acc_train')
        plt.subplot(2, 1, 2)
        plt.xlabel('Epoch')
        plt.ylabel('acc_test')
        plt.plot(range(epochs), acc_test_record, label='acc_test', color='red')
        plt.show()
