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

'''
if __name__ == '__main__':
    fileHandle = open('label_key_dict.txt', 'wb')
    # 图片标签与key_word对应关系
    l1 = ['江海湖', '鱼游水', '鱼']
    l2 = ['春花', '秋落', '春红', '春风', '秋凉', '枝', '桃']
    l3 = ['杯', '盘']
    l4 = ['桃', '枝']
    l5 = ['金', '盘']
    l6 = ['飞', '飞高']
    l7 = ['山', '野']
    l8 = ['楼', '高', '阁']
    l9 = ['树', '林', '枝']
    l10 = ['车', '车路']
    label_key_dict = {'beaver': l1, 'dolphin': l1, 'otter': l1, 'seal': l1, 'aquarium_fish': l1, 'flatfish': l1,
                      'ray': l1, 'shark': l1, 'trout': l1, 'whale': l1, 'orchid': l2, 'poppie': l2, 'rose': l2,
                      'sunflower': l2, 'tulip': l2, 'bottle': l3, 'bowl': l3, 'can': l3, 'cup': l3,
                      'plate': l3, 'apple': l4, 'mushroom': l4, 'orange': l4, 'pear': l4, 'sweet_pepper': l4,
                      'clock': l5, 'computer keyboard': l5, 'lamp': l5, 'telephone': l5, 'television': l5,
                      'bed': l5, 'chair': l5, 'couch': l5, 'table': l5, 'wardrobe': l5,
                      'bee': l6, 'beetle': l6, 'butterfly': l6, 'caterpillar': l6, 'cockroach': l6,
                      'bear': l7, 'leopard': l7, 'lion': l7, 'tiger': l7, 'wolf': l7,
                      'bridge': ['桥'], 'castle': l8, 'house': l8, 'road': ['路', '道'], 'skyscraper': l8,
                      'cloud': ['云'], 'forest': ['林'], 'mountain': ['山'], 'plain': ['平'], 'sea': ['海'],
                      'camel': ['沙'], 'cattle': l7, 'chimpanzee': l7, 'elephant': l7, 'kangaroo': l7,
                      'fox,': l7, 'porcupine': l7, 'possum': l7, 'raccoon': l7, 'skunk': l7,
                      'crab': ['鱼'], 'lobster': ['鱼'], 'snail': ['地'], 'spider': ['地'], 'worm': ['地'],
                      'baby': ['儿', '小子'], 'man': ['夫'], 'womam': ['女'], 'boy': ['郎'], 'girl': ['女'],
                      'crocodile': l7, 'dinosaur': l7, 'lizard': l7, 'snake': l7, 'turtle': l7,
                      'hamster': l7, 'mouse': l7, 'rabbit': l7, 'shrew': l7, 'squirrel': l7,
                      'maple_tree': l9, 'oak_tree': l9, 'palm_tree': l9, 'pine_tree': l9, 'willow_tree': l9,
                      'bicycle': l10, 'bus': l10, 'motorcycle': l10, 'pickup truck': l10, 'train': l10,
                      'lawn_mower': l10, 'rocket': l10, 'streetcar': l10, 'tank': l10, 'tractor': l10}
    pickle.dump(label_key_dict, fileHandle)
    fileHandle.close()'''

# 生成几首诗
generate_totalNum = 5





