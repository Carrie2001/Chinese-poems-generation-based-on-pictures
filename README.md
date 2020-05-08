# Chinese-poems-generation-based-on-pictures
Generation of five character ancient poetry based on given pictures(基于图片生成五言古诗)
# We will introduce the use of each PY
cifar100vgg.py is used to process image files and get related labels. The label will then correspond to key words to generate poems.
data.py is used to handle related poems txt file, generate corresponding words_vocabulary and word_vec_dict and so on.
json_handle.py is a program to select five character quatrain from the data_collection in https://github.com/chinese-poetry/chinese-poetry
parameter.py (just like its name)
rnn_model.py contains a s2s model based on attention mechanism
train.py contains train, generate and api with image
word2vec.py trains the word_vector based on gensim
