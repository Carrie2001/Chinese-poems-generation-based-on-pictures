# coding: UTF-8
# 模型采用基于attention的s2s模型

from parameter import *
from wordvec import *


class model:
    def __init__(self, traindata, size):
        # 实例化数据集处理的对象
        self.traindata = traindata
        # 输入的诗句，输出的诗句，关键词标签
        self.gtX = tf.placeholder(tf.int32, shape=[size, None])  # input
        self.gtY = tf.placeholder(tf.int32, shape=[size, None])  # output
        self.gtZ = tf.placeholder(tf.int32, shape=[size, None])  # key word
        self.rnn_lstm(self.traindata.wordNum + 1)

    def rnn_lstm(self, wordNum, hidden_units=100, layers=2):
        with tf.variable_scope("embedding"):
            # 将字向量嵌入gtx、gtz
            # 用之前训练好的字向量初始化
            embedding = tf.get_variable("embedding", initializer=word_vectors, dtype=tf.float32)
            self.inputbatch = tf.nn.embedding_lookup(embedding, self.gtX)
            self.input_z_batch = tf.nn.embedding_lookup(embedding, self.gtZ)

        with tf.variable_scope("encoder"):
            # 把关键词向量用lstm网络处理后成一个向量放入decoder的rnn网络输入中
            self.basicCell2 = tf.contrib.rnn.BasicLSTMCell(hidden_units, state_is_tuple=True)
            self.stackCell2 = tf.contrib.rnn.MultiRNNCell([self.basicCell2] * layers)
            self.initState2 = self.stackCell2.zero_state(np.shape(self.gtZ)[0], tf.float32)
            self.outputs2, self.finalState2 = tf.nn.dynamic_rnn(self.stackCell2, self.input_z_batch,
                                                              initial_state=self.initState2)
            self.c = tf.reduce_sum(self.outputs2, axis=1)
            self.c1 = tf.expand_dims(self.c, 1)
            self.input_batch = tf.add(self.inputbatch, self.c1)

        # 以下全为decoder
        with tf.variable_scope("cell"):
            # 两层rnn-lstm循环网络
            self.basicCell = tf.contrib.rnn.BasicLSTMCell(hidden_units, state_is_tuple=True)
            self.stackCell = tf.contrib.rnn.MultiRNNCell([self.basicCell] * layers)
            self.initState = self.stackCell.zero_state(np.shape(self.gtX)[0], tf.float32)
            self.outputs, self.finalState = tf.nn.dynamic_rnn(self.stackCell, self.input_batch,
                                                              initial_state=self.initState)
            self.outputs = tf.reshape(self.outputs, [-1, hidden_units])

        with tf.name_scope("softmax"):
            # 将最后处理的向量进行概率求解，映射到整个词表上
            w = tf.get_variable("w", [hidden_units, wordNum])
            b = tf.get_variable("b", [wordNum])
            self.logits = tf.matmul(self.outputs, w) + b
            self.probs = tf.nn.softmax(self.logits)

        with tf.name_scope('loss'):
            # 先将target扩展一维，在于logitsoftmax后进行交叉熵计算损失值
            self.targets = tf.reshape(self.gtY, [-1])
            self.loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([self.logits], [self.targets],
                                                                      [tf.ones_like(self.targets, dtype=tf.float32)])
            self.globalStep = tf.Variable(0, trainable=False)
            self.addGlobalStep = self.globalStep.assign_add(1)
            self.cost = tf.reduce_mean(self.loss)

        with tf.name_scope('op'):
            # 梯度裁剪进行梯度函数计算
            trainableVariables = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainableVariables),
                                              5)  # prevent loss divergence caused by gradient explosion
            # 指数衰减学习率
            learningRate = tf.train.exponential_decay(learningRateBase, global_step=self.globalStep,
                                                      decay_steps=learningRateDecayStep,
                                                      decay_rate=learningRateDecayRate)
            optimizer = tf.train.AdamOptimizer(learningRate)
            self.trainOP = optimizer.apply_gradients(zip(grads, trainableVariables))
