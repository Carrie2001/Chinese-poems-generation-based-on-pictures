import tensorflow as tf
import numpy as np
import argparse
import os
import random
import time
import collections
import pickle

# 一些基本的数据参数
batch_size = 64

learningRateBase = 0.001
learningRateDecayStep = 1000
learningRateDecayRate = 0.95

epochNum = 100                   # train epoch
generateNum = 1                  # number of generated poems per time

type = "poetry"                   # dataset to use, shijing, songci, etc
trainPoems = "./dataset/" + type + "/" + type + ".txt" # training file location
checkpointsPath = "checkpoints_key/" # checkpoints location

saveStep = 200                   # save model every savestep



# evaluate
trainRatio = 0.8                    # train percentage
evaluateCheckpointsPath = "./checkpoints/evaluate"


charvec_path = 'charvec_save'


# 生成输入数据的保存路径
path_x = 'batch/x.npy'
path_y = 'batch/y.npy'
path_z = 'batch/z.npy'


# 图片与关键词的对应词典
fileHandle = open('label_key_dict.txt', 'rb')
label_key_dict = pickle.load(fileHandle)
fileHandle.close()


# 生成几首诗
generate_totalNum = 5





