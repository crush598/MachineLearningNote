# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 3:11 PM
# @Author  : Hush
# @Email   : crush@tju.edu.cn

from sklearn import datasets
from sklearn.model_selection import train_test_split
import binary_tree as DecisionTree


def data_process():
    # 加载乳腺癌数据集
    cancers = datasets.load_breast_cancer()
    feature_names = cancers.feature_names
    feature_names = list(feature_names)
    X = cancers.data
    y = cancers.target
    # print(X.shape, y.shape)
    # for i in range(len(feature_names)):
    #     plt.scatter(X[:,i],y,s=20)
    #     plt.title(feature_names[i]+ f"{i}")
    #     plt.show()
    # X = X[:, [0, 2, 3, 7, 8, 20, 21, 22, 23, 27]]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
    return x_train, x_test, y_train, y_test, feature_names




if __name__ == "__main__":
    x_train, x_test, y_train, y_test, features = data_process()
    num_samples, num_features = x_train.shape
    best_max_depth = num_features
    best_min_samples_split = 10
    best_val_acc = None

    dt = DecisionTree.DecisionTree(features=features, max_depth=best_max_depth, min_samples_split=best_min_samples_split)
    dt.fit(x_train, y_train)
    acc = dt.predict(x_test,y_test)


    print(f"Best max_depth: {best_max_depth}, best min_samples_split: {best_min_samples_split}")
    print(f"Best validation set accuracy: {acc}")
