# -*- coding: utf-8 -*-
# @Time    : 2022/10/12 8:32 AM
# @Author  : Hush
# @Email   : crush@tju.edu.cn
from collections import OrderedDict
from pprint import pprint
import numpy as np
import pandas as pd


class Apriori:
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

    def support(self, names):
        new_data = self.data[names]
        count = 0
        for i in range(self.total):
            if list(new_data.loc[i]) == [1] * len(names):
                count = count + 1
        return count

    def find_rules(self):
        self.strat_set()
        rules = [self.items]
        while rules[-1]:
            rules.append({})
            K_List = sorted(rules[-2].keys())
            length = len(K_List)
            for i in range(length):
                for j in range(i + 1, length):
                    name1 = K_List[i].split(",")
                    name2 = K_List[j].split(",")
                    if name1[:-1] == name2[:-1]:
                        name1.append(name2[-1])
                        support = self.support(name1)
                        if support >= self.min_support:
                            rules[-1][",".join(name1)] = support
        # 遍历每一个频繁项，计算置信度
        # print(rules)
        result = {}
        for n, rule in enumerate(rules[1:]):  # 对于所有的k，遍历k频繁项
            for r, v in rule.items():  # 遍历所有的k频繁项
                names = r.split(",")
                for i, _ in enumerate(names):  # 遍历所有的排列，即(A,B,C)究竟是 A,B -> C 还是 A,B -> C 还是 A,B -> C ？
                    x = names[:i] + names[i + 1:]
                    # print(x,names[i])
                    confidence = v / self.support(x)  # 不同排列的置信度
                    # print(confidence)
                    if confidence >= self.min_confidence:  # 如果某种排列的置信度足够大，那么就加入到结果
                        result[",".join(x) + "=>>" + names[i]] = (v, confidence)
        return result

    def pretty_print_frequent_patterns(self, frequent_patterns):
        frequent_patterns = OrderedDict(sorted(frequent_patterns.items(), key=lambda x: x[1], reverse=True))
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
    print("The Result Of Apriori:")
    AP = Apriori(path, 8, 0.8)
    result = AP.find_rules()
    AP.pretty_print_frequent_patterns(result)
