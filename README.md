# Chinese-poems-generation-based-on-pictures
Generation of five character ancient poetry based on given pictures(基于图片生成五言古诗)

You also need to put the following linked files in the same directory(还需把如下链接的文件放入同一个目录下)：
https://disk.pku.edu.cn:443/link/04FD3AA91645117E8477FE32D15FE31C
有效期限：2024-06-14 23:59


# We will introduce the use of each PY
cifar100vgg.py is used to process image files and get related labels. The label will then correspond to key words to generate poems.

data.py is used to handle related poems txt file, generate corresponding words_vocabulary and word_vec_dict and so on.

json_handle.py is a program to select five character quatrain from the data_collection in https://github.com/chinese-poetry/chinese-poetry

parameter.py (just like its name)

rnn_model.py contains a s2s model based on attention mechanism

train.py contains train, generate and api with image

word2vec.py trains the word_vector based on gensim

# Examples
![image](https://github.com/Gold-Sea/Chinese-poems-generation-based-on-pictures/blob/master/readme_pictures/fs.jpg)
老墨天平色，秋风正峻津。
亭家人不尽，花落影斜喧。
