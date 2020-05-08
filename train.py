# coding: UTF-8
from parameter import *
from rnn_model import *
from wordvec import*


def train(traindata, reload = True):
    # 初始化计算图
    lstm_model = model(traindata, batch_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # 进行模型的加载
        if not os.path.exists(checkpointsPath):
            os.mkdir(checkpointsPath)

        if reload:
            checkPoint = tf.train.get_checkpoint_state(checkpointsPath)
            # 如果有模型就加载，没有就从头训练
            if checkPoint and checkPoint.model_checkpoint_path:
                saver.restore(sess, checkPoint.model_checkpoint_path)
                print("restored %s" % checkPoint.model_checkpoint_path)
            else:
                print("no checkpoint found!")
        # 开始训练
        for epoch in range(epochNum):
            X, Y, Z = generateBatch()
            epochSteps = len(X)  # equal to batch
            for step, (x, y, z) in enumerate(zip(X, Y, Z)):
                a, loss, gStep = sess.run([lstm_model.trainOP, lstm_model.cost, lstm_model.addGlobalStep],
                                          feed_dict={lstm_model.gtX: x, lstm_model.gtY: y, lstm_model.gtZ:z})
                print("epoch: %d, steps: %d/%d, loss: %3f" % (epoch + 1, step + 1, epochSteps, loss))
                if gStep % saveStep == saveStep - 1:  # prevent save at the beginning
                    print("save model")
                    saver.save(sess, os.path.join(checkpointsPath, type), global_step=gStep)


# 根据映射的概率提取相应的字
def probsToWord(weights, words):
    # ratio是任取的，在1~0均匀分布，出现在概率大的区间的概率也会很大
    prefixSum = np.cumsum(weights)  # prefix sum
    ratio = np.random.rand(1)
    index = np.searchsorted(prefixSum, ratio * prefixSum[-1])
    if index[0] >= len(words):
        index[0] = np.random.rand(len(words) - 1)[0]
    index = index[0]
    return words[index]


# 检查是否为五言绝句，删除非五言绝句
def examine_poems(poems, generate_num):
    wrong_id = []  # 错误诗的序号
    for i in range(generate_num):
        poem = poems[i]
        if len(poem) != 28:
            wrong_id.append(i)
            continue
        for j in [x * 7 for x in range(4)]:
            for k in range(5):
                if poem[j + k] in ['，', '。', '\n']:  # 是否是字
                    wrong_id.append(i)
                    continue
            separate = {0: '，', 1: '。'}
            if poem[j + 5] != separate[j // 7 % 2]:  # 标点是否正确
                wrong_id.append(i)
                continue
            if poem[j + 6] != '\n':  # 是否正确换行
                wrong_id.append(i)
    right_poems = []
    for i in range(generate_num):
        if i in wrong_id:
            continue
        right_poems.append(poems[i])
    generate_num = len(right_poems)
    return right_poems, generate_num


# characters是关键字对应数字列表
def generate(traindata, characters):
    print("genrating...")
    tf.reset_default_graph()
    # 初始化计算图
    lstm_model = model(traindata, 1)
    # 检索关键字
    for i in characters:
        if not (i in poem_Data.key_word_num):
            print('输入的关键词不在关键词库中')
            return None
    _characters = np.array([characters])
    with tf.Session() as sess:
        # 加载训练好的模型
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        checkPoint = tf.train.get_checkpoint_state(checkpointsPath)
        if checkPoint and checkPoint.model_checkpoint_path:
            saver.restore(sess, checkPoint.model_checkpoint_path)
            print("restored %s" % checkPoint.model_checkpoint_path)
        else:
            print("no checkpoint found!")
            exit(1)
        # 先计算出第一个状态
        state = sess.run(lstm_model.stackCell.zero_state(1, tf.float32))
        # x为feed
        x = [[lstm_model.traindata.word_ID['[']]]
        poems = []
        for i in range(generateNum):
            # 根据上一个状态、attention与这时的输入计算出输出的概率，由此找到对应的词，进行生成诗句
            probs1, state = sess.run([lstm_model.probs, lstm_model.finalState], feed_dict={lstm_model.gtX: x,
                                                                                           lstm_model.gtZ: _characters,
                                                                                           lstm_model.initState: state})
            word = probsToWord(probs1, lstm_model.traindata.word_vca)
            poem = ''
            sentenceNum = 0
            while word not in [' ', ']']:
                poem += word
                if word in ['。', '？', '！', '，']:
                    sentenceNum += 1
                    if sentenceNum % 2 == 0:
                        poem += '\n'
                x = [[lstm_model.traindata.word_ID[word]]]
                # print(word)
                probs2, state = sess.run([lstm_model.probs, lstm_model.finalState], feed_dict={
                    lstm_model.gtX: x, lstm_model.gtZ: _characters, lstm_model.initState: state})
                word = probsToWord(probs2, lstm_model.traindata.word_vca)
            poems.append(poem)
        '''for word in characters:
            if lstm_model.traindata.word_ID.get(word) == None:
                print("不认识这个字, 抱歉")
                exit(0)
            flag = -flag
            while word not in [']', '，', '。', ' ', '？', '！']:
                poem += word
                x = np.array([[lstm_model.traindata.word_ID[word]]])
                probs2, state = sess.run([lstm_model.probs, lstm_model.finalState], feed_dict={lstm_model.gtX: x, lstm_model.initState: state})
                word = probsToWord(probs2, lstm_model.traindata.word_vca)

            poem += endSign[flag]
            # keep the context, state must be updated
            if endSign[flag] == '。':
                probs2, state = sess.run([lstm_model.probs, lstm_model.finalState],
                                            feed_dict={lstm_model.gtX: np.array([[lstm_model.traindata.word_ID["。"]]]),
                                                        lstm_model.initState: state})
                poem += '\n'
            else:
                probs2, state = sess.run([lstm_model.probs, lstm_model.finalState],
                                             feed_dict={lstm_model.gtX: np.array([[lstm_model.traindata.word_ID["，"]]]),
                                                        lstm_model.initState: state})'''
        return poem


# 该函数根据得到的图片标签生成有关的诗句
def label_poem(label):
    # 图片对应的关键字
    key_list = label_key_dict[label]
    rand_np = np.random.randint(len(key_list), size=5)
    poem_list = []
    # 生成五首诗句并存入列表最后return
    for i in range(generate_totalNum):
        characters = list(key_list[rand_np[i]])
        key_num = []
        # 拓展关键字
        _characters = charvec().gene_simi_chars(characters)
        for i in _characters:
            key_num.append(poem_Data.word_ID[i])
            print(i, end='')
        print()
        poems = generate(poem_Data, key_num)
        p = list(poems)
        p.insert(6, '\n')
        p.insert(20, '\n')
        poems = ''.join(p)
        poem_list.append(poems)
    poem_list, generate_num = examine_poems(poem_list, generate_totalNum)
    print("生成%d首诗\n" % generate_num)
    for i in range(generate_num):
        print(poem_list[i])
    return poem_list
