# coding: UTF-8
# 本程序用gensim训练字向量
from data import *
from parameter import *
from gensim import models
import numpy as np


class charvec:
    # 初始训练字向量的模型，如果已经训练过了就加载
    def __init__(self, istrain = False):
        if istrain:
            self.model = models.Word2Vec(poem_Data.poems, sg=1, size=100, window=5, min_count=2, negative=3, sample=0.001, hs=1,
                             workers=4)
            self.model.train(sentences=poem_Data.poems, total_examples=len(poem_Data.poems), epochs=20)
            self.model.save(charvec_path)
        else:
            self.model = models.Word2Vec.load(charvec_path)

    # 关键字补全（输入一个列表，返回一个列表，返回的列表是意思相近的关键字）
    def gene_simi_chars(self, characters):
        res = characters
        if len(characters) == 1:
            # 将字库排序，排序标准是与输入列表的相似度
            simi_list = self.model.most_similar(positive=characters, topn=4750)
            l = []
            count = 0
            # 拓展为四个关键字
            for i in range(4000):
                if simi_list[i][0] in poem_Data.key_word:
                    l.append(simi_list[i][0])
                    count += 1
                if count > 3:
                    break
            for i in range(3):
                res.append(l[i])
        elif len(characters) == 2:
            simi_list = self.model.most_similar(positive=characters, topn=4750)
            l = []
            count = 0
            for i in range(4000):
                if simi_list[i][0] in poem_Data.key_word:
                    l.append(simi_list[i][0])
                    count += 1
                if count > 2:
                    break
            for i in range(2):
                res.append(l[i])
        elif len(characters) == 3:
            simi_list = self.model.most_similar(positive=characters, topn=4750)
            l = []
            count = 0
            for i in range(4000):
                if simi_list[i][0] in poem_Data.key_word:
                    l.append(simi_list[i][0])
                    count += 1
                if count > 1:
                    break
            res.append(l[0])
        return res


# 整理字向量库并用来初始化embedding
c = charvec().model
word_vectors = np.zeros(shape=[len(poem_Data.word_ID) + 1, 100], dtype=np.float32)
l = list(poem_Data.word_ID.keys())
for i in range(len(poem_Data.word_ID)):
    tmp = c[l[i]]
    word_vectors[i] = tmp


# 生成训练用到的三个数据集
def generateBatch(istrain=True, reload=True):
    if istrain:
        poemsVector = poem_Data.trainVector
    else:
        poemsVector = poem_Data.testVector
    # 因为训练集数据加载的比较慢(5min左右),所以预先处理好用到时只需加载
    if reload:
        X = np.load(path_x)
        Y = np.load(path_y)
        Z = np.load(path_z)
        return X, Y, Z
    random.shuffle(poemsVector)
    batchNum = (len(poemsVector) - 1) // batch_size
    # X为输入诗句向量，Y为输出，Z为关键字的标签集
    X = []
    Y = []
    Z = []  # 关键词标签
    for i in range(batchNum):
        print(i)
        batch = poemsVector[i * batch_size: (i + 1) * batch_size]
        length = 26
        temp = np.zeros((batch_size, length), dtype=np.int32)
        temp2 = np.zeros((batch_size, 4), dtype=np.int32)
        for j in range(batch_size):
            temp[j] = batch[j]
            # 关键字的提取，找到与诗句与关键字库的交集，随机的选取四个，如果不足四个就进行关键字补全（慢的原因就是补全的效率较低）
            word_num = []
            words = []
            l = []
            # 取交集
            for i in temp[j]:
                if i in poem_Data.key_word_num:
                    word_num.append(i)
            for i in word_num:
                words.append(poem_Data.word_numtoID[i])
            # 不用补全
            if len(words) >= 4:
                ran = np.random.randint(len(words), size=4)
                for i in ran:
                    l.append(words[i])
            # 补全关键字
            else:
                if len(words) == 0:
                    words.append('不')
                l = charvec().gene_simi_chars(words)
            tmpint = []
            for i in l:
                tmpint.append(poem_Data.word_ID[i])
            temp2[j] = tmpint
        X.append(temp)
        Z.append(temp2)
        temp2 = np.copy(temp)
        temp2[:, :-1] = temp[:, 1:]
        # 保存数据集
        Y.append(temp2)
        x = np.array(X)
        y = np.array(Y)
        z = np.array(Z)
        np.save(path_x, x)
        np.save(path_y, y)
        np.save(path_z, z)
    return x, y, z


if __name__ == '__main__':
    c = charvec().gene_simi_chars(['妇'])
    print(c)

