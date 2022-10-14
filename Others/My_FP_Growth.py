# -*- coding: utf-8 -*-
# @Time    : 2022/10/12 8:32 AM
# @Author  : Hush
# @Email   : crush@tju.edu.cn

from pprint import pprint
import numpy as np
import pandas as pd
from collections import Counter, OrderedDict, defaultdict
from itertools import combinations
from typing import Union


def pythonic_count_items(table):
    all_items = [item for sublist in table for item in sublist]
    counter = Counter(all_items)
    return counter, len(table)


def fast_count_items(table):
    counter = defaultdict(int)

    for transaction in table:
        for item in transaction:
            counter[item] += 1

    return counter, len(table)


def sort_transactions(table, counter):
    return [sort_items(transaction, counter) for transaction in table]


def sort_items(transaction, counter):
    return sorted(transaction, key=lambda x: (counter[x], str(x)), reverse=True)


def delete_items_with_no_support(table: list[list[str]], counter: dict, min_support: int):
    counter = OrderedDict([(item, number) for item, number in counter.items() if number >= min_support])
    new_table = [[item for item in transaction if counter.get(item)] for transaction in table]
    return new_table, counter


def create_sorted_representation(frequent_patterns: dict, counter: dict) -> dict:
    frequent_patterns = {'; '.join(sort_items(list(name), counter)): value for name, value in frequent_patterns.items()}
    return frequent_patterns


class Node(object):
    # add slots to impede the creation of object __dict__ in order to reduce the memory footprint of each tree
    # comes with the cost of not being able to store attributes dynamically, but we don't need that anyway.
    __slots__ = 'parent', 'value', 'children', 'counter', 'singular'

    def __init__(self, parent, value):
        """
        This function creates a base node for a tree. Please use base_node.adopt() for child creation!

        :param parent: the parent node in the tree.
        :param value: the value of the current node.
        """

        # save the values
        self.parent = parent  # the parent node (to traverse the tree)
        self.value = value  # the actual value (item) of the node
        self.children = {}  # a dict of children, where we will save them with their names
        self.counter = 0  # the counter how many times the item exists in the tree
        self.singular = True  # value to store whether the tree is singular (only one branch)

    def adopt(self, value):
        """
        This function creates a child node with the given value and inserts it in the tree. It also takes care of the
        tracking, whether the tree is singular.

        :param value: the item name
        :return: the child node
        """

        # try to get the child from dict of children
        child = self.children.get(value)

        # create child and check singularity if we have more than one child after creation
        if child is None:

            # create child
            child = Node(self, value)

            # save child in dict of children
            self.children[value] = child

            # as tree can only go from singular -> not singular, we only need to check the children number if tree is
            # still singular.
            if self.singular and len(self.children) > 1:
                # set self and all others to not singular
                self.not_singular()

        # return the child node that has either been created or found in children dict
        return child

    def increment(self):
        """
        This function recursively increments the use counter for all nodes in one branch.

        :return: None
        """
        self.counter += 1
        if self.parent is not None:
            self.parent.increment()

    def not_singular(self):
        """
        This function recursively sets all parent nodes to not singular once it is called.

        :return: None
        """
        # as the tree can only go from singular -> not singular, the first if saves functions calls in case we encounter
        # the first already not singular part of tree
        if self.singular:

            # set self to not singular
            self.singular = False

            # iteratively go up the tree
            if self.parent is not None:
                self.parent.not_singular()

    def pretty_print(self, heading='1.', start_str=""):
        """
        This function creates a nicely formatted string of the current node, and it's children.

        :param heading: The starting heading for the representation.
        :param start_str: The appendix for every line of the tree.
        :return: String representation of the tree.
        """
        tabs = "\t" * (len(heading) // 2 - 1)
        start_str += f'{tabs}{heading} {self.value}: {self.counter}\n'

        # make comment if tree is singular
        if self.parent is None:
            start_str = start_str[:-1] + (' -> singular\n' if self.singular else ' -> not singular\n')

        for counter, child in enumerate(self.children.values(), start=1):
            start_str += child.pretty_print(heading=heading + f'{str(counter)}.')

        return start_str

    def __str__(self):
        """
        This function is a wrapper so self.pretty_print() is called once somebody attempts to print the node.

        :return: Formatted string representation of the tree.
        """
        return self.pretty_print()


def construct_tree(table: list[list[str]], start_node_name: tuple = None, min_support=0) -> [Node, dict, dict]:
    counter, number = fast_count_items(table)

    table, counter = delete_items_with_no_support(table, counter, min_support)

    head_table = {name: set() for name in counter}

    if start_node_name is None:
        table = sort_transactions(table, counter)

    base_node = Node(None, start_node_name)

    for transaction in table:

        current_node = base_node

        for item in transaction:
            current_node = current_node.adopt(item)

            head_table[item].update([current_node])

        current_node.increment()

    return base_node, head_table, counter


def count_frequent_patterns(table: list[Union[list, set]], condition: list = None, min_support=0,
                            frequent_patterns: dict = None) -> dict:
    # build the first tree
    tree, head_table, counter = construct_tree(table=table, start_node_name=condition, min_support=min_support)

    if frequent_patterns is None:
        frequent_patterns = defaultdict(int)

    if condition is None:
        condition = []

    if tree.singular:

        node_list = list(head_table.keys())

        for combination_length in range(1, len(node_list) + 1):

            combis = combinations(node_list, combination_length)
            for combi in combis:

                support = tree.counter
                for item in combi:
                    support = min(support, sum([node.counter for node in head_table[item]]))

                if tree.value is not None:
                    combi = tree.value + list(combi)

                frequent_patterns[frozenset(combi)] += support

    else:
        for item in head_table:

            conditional_table = []

            current_condition_support = 0
            for node in head_table[item]:

                node_counter = node.counter

                current_condition_support += node_counter

                transaction = []

                while node.parent.parent is not None:
                    transaction.insert(0, node.parent.value)

                    node = node.parent

                if transaction:
                    conditional_table += [transaction] * node_counter

            new_condition = condition + [item]

            frequent_patterns[frozenset(new_condition)] += current_condition_support

            _ = count_frequent_patterns(conditional_table,
                                        condition=new_condition,
                                        min_support=min_support,
                                        frequent_patterns=frequent_patterns)

    if not condition:
        frequent_patterns = create_sorted_representation(frequent_patterns, counter)
    return frequent_patterns


class FP_growth:
    def __init__(self, path, min_support, min_confidence):
        self.min_support = min_support  # 最小支持度
        self.min_confidence = min_confidence  # 最小置信度
        self.data = pd.read_csv(path)
        self.columns = self.data.columns
        self.total = self.data.shape[0]

    def strat_set(self):
        self.items = {}
        for i in self.columns:
            temp = 0
            for j in self.data[i]:
                if j == 1:
                    temp = temp + 1
            if temp >= self.min_support:
                self.items[i] = temp
        # print(self.items)

        temp = sorted(self.items.items(), key=lambda x: (-x[1], x[0]))
        self.rank_list = []
        for i in range(self.total):
            test = []
            for j in temp:
                if self.data[j[0]][i] == 1:
                    test.append(j[0])
            self.rank_list.append(test)
        # pprint(self.rank_list)

    def support(self, names):
        new_data = self.data[names]
        count = 0
        for i in range(self.total):
            if list(new_data.loc[i]) == [1] * len(names):
                count = count + 1
        return count

    def find_rules(self):
        self.strat_set()
        table = [set(transaction) for transaction in self.rank_list]
        rule = count_frequent_patterns(table, min_support=self.min_support)
        result = {}
        for r, v in rule.items():  # 遍历所有的k频繁项
            if r in self.items:
                pass
            else:
                names = r.split("; ")
                for i, _ in enumerate(names):  # 遍历所有的排列，即(A,B,C)究竟是 A,B -> C 还是 A,B -> C 还是 A,B -> C ？
                    x = names[:i] + names[i + 1:]
                    # print(x,names[i])
                    confidence = v / self.support(x)  # 不同排列的置信度
                    # print(confidence)
                    if confidence >= self.min_confidence:  # 如果某种排列的置信度足够大，那么就加入到结果
                        result[";".join(x) + "=>>" + names[i]] = (v, confidence)
        # print(result)
        return result

    def pretty_print_frequent_patterns(self, frequent_patterns):

        frequent_patterns = OrderedDict(sorted(frequent_patterns.items(), key=lambda x: x[1], reverse=True))
        # print(frequent_patterns)
        columns = ['Pattern', 'Support', 'Confidence (%)']

        max_length_names = 15
        max_length_counter = 10
        max_length_percent = 15

        filler_string = '|-' + '-' * max_length_names + '-+-' + '-' * max_length_counter + '-+-' \
                        + '-' * max_length_percent + '-|\n'
        print_string = filler_string
        print_string += f'| {columns[0]: <{max_length_names}} | ' \
                        f'{columns[1]:>{max_length_counter}} | ' \
                        f'{columns[2]:>{max_length_counter}} |\n'
        print_string += filler_string

        for name, counter in frequent_patterns.items():
            print_string += f'| {name: <{max_length_names - 2}} | ' \
                            f'{counter[0]: >{max_length_counter}} | ' \
                            f'{counter[1]:>{max_length_counter + 4}.2f} |\n'

        print_string += filler_string

        print(print_string)


if __name__ == "__main__":
    path = "./data/食品问卷.csv"

    print("The Result Of FP_growth: ")
    FP = FP_growth(path, 8, 0.8)
    result = FP.find_rules()
    FP.pretty_print_frequent_patterns(result)
