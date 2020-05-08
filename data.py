# coding: UTF-8
import os
import numpy as np
from parameter import*


#为五言绝句的版本
filename = '原始古诗数据.txt'


class poems_data:
    def __init__(self, isEvaluate = False):
        # 先打开txt文件
        self.poems = []
        file = open(filename, "r", encoding='utf-8')
        for line in file:  # every line is a poem
            # print(line)
            l = line.strip().split(":")  # get title and poem
            poem = l[4]
            poem = poem.replace(' ', '')
            # 把一些含有注释的诗句删去
            if '_' in poem or '《' in poem or '[' in poem or '(' in poem or '（' in poem:
                continue
            # 算上标点符号为4*6=24
            if len(poem) != 24:  # filter poem
                continue
            poem = '[' + poem + ']'  # add start and end signs
            self.poems.append(poem)
        # 建立字表
        self.word_dict = {}
        for poem in self.poems:
            for word in poem:
                if word not in self.word_dict:
                    self.word_dict[word] = 1
                else:
                    self.word_dict[word] += 1
        # 删除一些生僻字
        self.word_erase = []
        for key in self.word_dict.keys():
            if self.word_dict[key] <= 2:
                self.word_erase.append(key)
        for key in self.word_erase:
            del self.word_dict[key]

        self.word_dict = sorted(self.word_dict.items(), key=lambda d: d[1], reverse=True)
        # 建立关键词库
        self.key_word = []
        for i in range(4, 604, 1):
            self.key_word.append(self.word_dict[i][0])
        self.word_vca, ta = zip(*self.word_dict)
        # 统计字表的大小
        self.wordNum = len(self.word_vca)
        # 字与one—hot向量对应的关系
        self.word_ID = dict(zip(self.word_vca, range(len(self.word_vca))))
        self.word_numtoID = dict(zip(range(len(self.word_vca)), self.word_vca))
        self.wordTOIDFun = lambda A: self.word_ID.get(A, len(self.word_vca))
        # 诗句向量
        self.poemsVector = [([self.wordTOIDFun(word) for word in poem]) for poem in self.poems]
        self.key_word_num = []
        for i in self.key_word:
            self.key_word_num.append(self.word_ID[i])
        # 是否是需要进行评估，目前评估函数没有写
        if isEvaluate:
            self.trainVector = self.poemsVector[:int(len(self.poemsVector) * trainRatio)]
            self.testVector = self.poemsVector[int(len(self.poemsVector) * trainRatio):]
        else:
            self.trainVector = self.poemsVector
            self.testVector = []


poem_Data = poems_data(False)

'''
# 把关键词写入txt方便查看
with open('关键词.txt', 'w', encoding='utf-8')as file:
    for i in poem_Data.key_word:
        file.write(i)
        file.write('\n')'''


