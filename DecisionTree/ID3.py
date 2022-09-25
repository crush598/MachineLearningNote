# -*- coding: utf-8 -*-
# @Time    : 2022/9/15 11:05 PM
# @Author  : Hush
# @Email   : crush@tju.edu.cn

import numpy as np
import pandas as pd


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        """
                Class for storing Decision Tree as a binary-tree
                Inputs:
                - feature: Name of the feature based on which this node is split
                - threshold: The threshold used for splitting this subtree
                - left: left Child of this node
                - right: Right child of this node
                - value: Predicted value for this node (if it is a leaf node)
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, features=None, max_depth=None, min_samples_split=2):
        """
            Class for implementing Decision Tree
            Attributes:
            - max_depth: int
                The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until
                all leaves contain less than min_samples_split samples.
            - min_num_samples: int
                The minimum number of samples required to split an internal node
            - root: Node
                Root node of the tree; set after calling fit.
        """
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.features = list(features)
        self.used_features_indices = [-1]

    def is_splitting_finished(self, depth, num_class_labels, num_samples):
        """
            Criteria for continuing or finishing splitting a node
            Inputs:
            - depth: depth of the tree so far
            - num_class_labels: number of unique class labels in the node
            - num_samples: number of samples in the node
            :return: bool
        """
        if depth >= self.max_depth or num_class_labels == 1 or num_samples < self.min_samples_split:
            return True
        return False

    def split(self, X, y, feature, threshold):
        """
            Splitting X and y based on value of feature with respect to threshold;
            i.e., if x_i[feature] <= threshold, x_i and y_i belong to X_left and y_left.
            Inputs:
            - X: Array of shape (N, D) (number of samples and number of features respectively), samples
            - y: Array of shape (N,), labels
            - feature: Name of the the feature based on which split is done
            - threshold: Threshold of splitting
            :return: X_left, X_right, y_left, y_right
        """
        feature_index = self.features.index(feature)
        ind_left = np.where(X[:, feature_index] <= threshold)[0]
        ind_right = np.where(X[:, feature_index] > threshold)[0]
        X_left = X[ind_left]
        X_right = X[ind_right]
        y_left = y[ind_left]
        y_right = y[ind_right]
        return X_left, X_right, y_left, y_right

    def entropy(self, y):
        """
            Computing entropy of input vector
            - y: Array of shape (N,), labels
            :return: entropy of y
        """
        unique_value, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -(probabilities * np.log2(probabilities)).sum()
        return entropy

    def information_gain(self, X, y, feature, threshold):
        """
            Returns information gain of splitting data with feature and threshold.
            Hint! use entropy of y, y_left and y_right.
        """
        parent_entropy = self.entropy(y)
        X_left, X_right, y_left, y_right = self.split(X, y, feature, threshold)
        if len(y_left) == 0 or len(y_right) == 0:
            return 0
        left_entropy = (len(y_left) / len(y)) * self.entropy(y_left)
        right_entropy = (len(y_right) / len(y)) * self.entropy(y_right)
        information_gain = parent_entropy - left_entropy - right_entropy
        return information_gain

    def best_split(self, X, y):
        """
            Used for finding best feature and best threshold for splitting
            Inputs:
            - X: Array of shape (N, D), samples
            - y: Array of shape (N,), labels
            :return:
        """
        best_feature = None
        best_threshold = None
        best_ig = None
        best_feature_index = None
        shuffled_indices = np.random.permutation(
            np.arange(len(self.features)))  # todo: You'd better use a random permutation of features 0 to D-1
        for index in shuffled_indices:
            if index not in self.used_features_indices:
                thresholds = np.unique(
                    X[:, index])  # todo: use unique values in this feature as candidates for best threshold
                for threshold in thresholds:
                    ig = self.information_gain(X, y, self.features[index], threshold)
                    if best_ig is None or ig > best_ig:
                        best_ig = ig
                        best_threshold = threshold
                        best_feature = self.features[index]
                        best_feature_index = index
        self.used_features_indices.append(best_feature_index)
        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0):
        """
            Recursive function for building Decision Tree.
            - X: Array of shape (N, D), samples
            - y: Array of shape (N,), labels
            - depth: depth of tree so far
            :return: root node of subtree
        """
        num_samples, _ = X.shape
        num_class_labels = len(np.unique(y))
        if self.is_splitting_finished(depth, num_class_labels, num_samples):
            one_count = len(np.where(y[:] == 1)[0])
            zero_count = len(np.where(y[:] == 0)[0])
            label = 1 if one_count > zero_count else 0
            return Node(value=label)
        best_feature, best_threshold = self.best_split(X, y)
        X_left, X_right, y_left, y_right = self.split(X, y, best_feature, best_threshold)
        left_subtree = self.build_tree(X_left, y_left, depth + 1)
        right_subtree = self.build_tree(X_right, y_right, depth + 1)
        # print(best_feature)
        return Node(best_feature, best_threshold, left_subtree, right_subtree)

    def fit(self, X, y):
        """
            Builds Decision Tree and sets root node
            - X: Array of shape (N, D), samples
            - y: Array of shape (N,), labels
        """
        self.root = self.build_tree(X, y)
        # for index in self.used_features_indices[1:]:
        # print(self.features[index])
        # print(self.used_features_indices)
        """
        ## 层序遍历 node ,并输出
        res = []
        queue = [self.root]
        while queue:
            res.append([node.feature for node in queue])
            temp = []
            for node in queue:
                if node.left:
                    temp.append(node.left)
                if node.right:
                    temp.append(node.right)
            queue = temp
        print(res)
        """
        return

    def predict(self, X):
        """
            Returns predicted labels for samples in X.
            :param X: Array of shape (N, D), samples
            :return: predicted labels
        """
        predicted_labels = []
        for row in X:
            predicted_labels.append(self.get_label(row, self.root))
        return np.array(predicted_labels)

    def get_label(self, row, node):
        if node.is_leaf():
            return node.value
        if row[self.features.index(node.feature)] <= node.threshold:
            return self.get_label(row, node.left)
        else:
            return self.get_label(row, node.right)
