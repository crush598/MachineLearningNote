# -*- coding: utf-8 -*-
# @Time    : 2022/10/26 08:50
# @Author  : Hush
# @Email   : crush@tju.edu.cn


import numpy as np
from collections import Counter
from treelib import Tree


class DecisionTree:
    def __init__(self, features, max_depth, min_samples_split):
        self.tree = Tree()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.features = list(features)
        self.used_features_indices = [-1]
        self.count = 0

    def entropy(self, y):
        unique_value, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -(probabilities * np.log2(probabilities)).sum()
        return entropy

    def information_gain(self, X, y, feature, threshold):
        parent_entropy = self.entropy(y)
        X_left, X_right, y_left, y_right = self.split(X, y, feature, threshold)
        if len(y_left) == 0 or len(y_right) == 0:
            return 0
        left_entropy = (len(y_left) / len(y)) * self.entropy(y_left)
        right_entropy = (len(y_right) / len(y)) * self.entropy(y_right)
        information_gain = parent_entropy - left_entropy - right_entropy
        return information_gain

    def best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_ig = None
        best_feature_index = None
        shuffled_indices = np.random.permutation(
            np.arange(len(self.features)))  # todo: You'd better use a random permutation of features 0 to D-1
        for index in shuffled_indices:
            if index not in self.used_features_indices:
                thresholds = np.unique(X[:, index])  # todo: use unique values in this feature as candidates for
                for threshold in thresholds:
                    ig = self.information_gain(X, y, self.features[index], threshold)
                    if best_ig is None or ig > best_ig:
                        best_ig = ig
                        best_threshold = threshold
                        best_feature = self.features[index]
                        best_feature_index = index
        self.used_features_indices.append(best_feature_index)
        return best_feature, best_threshold

    def split(self, X, y, best_feature, best_threshold):

        feature_index = self.features.index(best_feature)
        ind_left = np.where(X[:, feature_index] <= best_threshold)[0]
        ind_right = np.where(X[:, feature_index] > best_threshold)[0]
        return X[ind_left], X[ind_right], y[ind_left], y[ind_right]

    def is_split_stop(self, num_class_lable, num_samples):
        if self.used_features_indices[-1] is None or self.tree.depth() >= self.max_depth or num_class_lable == 1 or num_samples < self.min_samples_split:
            return True
        return False

    def build_tree(self, X, y, parent=None):
        num_samples = X.shape[0]
        num_class_lable = len(np.unique(y))
        best_feature, best_threshold = self.best_split(X, y)
        if self.is_split_stop(num_class_lable, num_samples):
            class_ = str(max(Counter(y)))
            self.tree.create_node(tag=class_, identifier=self.count, parent=parent,
                                  data=int(class_))
            self.count = self.count + 1
            return
        self.tree.create_node(tag=best_feature, identifier=self.count, parent=parent,
                              data=best_threshold)
        parent = self.count
        self.count = self.count + 1
        X_left, X_right, y_left, y_right = self.split(X, y, best_feature, best_threshold)
        self.build_tree(X_left, y_left, parent)
        self.build_tree(X_right, y_right, parent)
        return

    def fit(self, X, y):
        self.build_tree(X, y)
        self.tree.show()
        self.tree.save2file("Multiclass")

    def predict(self, X,y):
        y_hat = []
        for row in X:
            y_hat.append(self.get_label(row, self.tree.get_node(0)))
        acc = len(np.where(y == y_hat)[0]) / len(y) * 100
        # print(acc)
        return acc

    def get_label(self, row, node):
        if node.is_leaf():
            return node.data
        next_node = self.tree.children(node.identifier)
        if row[self.features.index(node.tag)] <= node.data:
            return self.get_label(row, next_node[0])
        else:
            return self.get_label(row,next_node[1])

from sklearn import datasets
from sklearn.model_selection import train_test_split


def data_process():
    # 加载乳腺癌数据集
    cancers = datasets.load_wine()
    feature_names = cancers.feature_names
    feature_names = list(feature_names)

    X = cancers.data
    y = cancers.target

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
    return x_train, x_test, y_train, y_test, feature_names




if __name__ == "__main__":
    x_train, x_test, y_train, y_test, features = data_process()
    num_samples, num_features = x_train.shape
    best_max_depth = num_features
    best_min_samples_split = 4
    best_val_acc = None

    dt = DecisionTree(features=features, max_depth=best_max_depth, min_samples_split=best_min_samples_split)
    dt.fit(x_train, y_train)
    acc = dt.predict(x_test,y_test)


    print(f"Best max_depth: {best_max_depth}, best min_samples_split: {best_min_samples_split}")
    print(f"Best validation set accuracy: {acc}")