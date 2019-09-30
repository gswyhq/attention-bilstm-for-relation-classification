#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 19:20:18 2018

@author: himanshu
https://raw.githubusercontent.com/0dust/Sentiment-Analysis-with-Attention/master/sentiment.py
"""

import os
import numpy as np
import pickle
# from nltk.corpus import stopwords
# from nltk.stem import SnowballStemmer
from keras.utils import to_categorical
import gensim
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
#from keras import initializations
from keras.callbacks import TensorBoard

from make_model import make_model
from utils import read_data, preprocess_text
from evaluate import evaluate
from config import DEV_FILE, TRAIN_FILE, EMB_FILE, MODEL_PATH, LOG_PATH, TOKENIZER_FILE

from config import MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, CHECKPOINT_FILE

flags = tf.app.flags
flags.DEFINE_boolean("train",       False,      "是否训练(默认：否)")
flags.DEFINE_string("log_path",     LOG_PATH,    "日志文件路径")
flags.DEFINE_string("model_path",    MODEL_PATH,     "模型文件路径")
flags.DEFINE_string("emb_file",     EMB_FILE,  "词向量文件路径")
flags.DEFINE_string("train_file",   TRAIN_FILE,  "训练文件路径")
flags.DEFINE_string("dev_file",     DEV_FILE,    "校验文件路径")

FLAGS = tf.app.flags.FLAGS

tokenizer_name = os.path.join(FLAGS.model_path, TOKENIZER_FILE)

# act = 'relu'

def load_word_vectors(file_path):
    """
    :return: words[], np.array(vectors)
    """

    print('Loading', file_path)
    if 'Tencent_AILab_ChineseEmbedding' in file_path:
        wv_from_text = KeyedVectors.load_word2vec_format(file_path, binary=False)
        words = wv_from_text.wv.index2word
        vectors = np.array([wv_from_text.get_vector(word) for word in words], dtype='float32', copy=False)
    else:
        model = gensim.models.Word2Vec.load(file_path)  # 加载词向量模型
        words = model.wv.index2word
        vector_size = model.vector_size
        vectors = [model.wv.get_vector(word) for word in words]
        vectors = np.array(vectors, dtype='float32', copy=False)
    return words, vectors

def train():

    ##################################################
    ## forming sequeces to feed into the network.
    ##################################################

    # 词向量对应的词列表，及对应词向量
    words, vectors = load_word_vectors(file_path=FLAGS.emb_file)
    embedding_index = {w: vec for w, vec in zip(words, vectors)}

    EMBEDDING_DIM = len(vectors[0])
    print('Indexed the word vectors')
    print('Found %s word vectors.' % len(embedding_index))

    train_data = read_data(FLAGS.train_file)
    test_data = read_data(FLAGS.dev_file) # [['怎么解释ALTHAUN', '定义']]

    # "id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate"
    raw_train_comments = [t[0] for t in train_data]
    raw_test_comments = [t[0] for t in test_data]
    classes_to_predict = list(set(t[1] for t in train_data) | set(t[1] for t in test_data))
    predicate_index = {predicate: _index for _index, predicate in enumerate(classes_to_predict)}

    predicate_categ = to_categorical(list(predicate_index.values()))

    predicate_label = {predicate: predicate_categ[_index] for predicate, _index in predicate_index.items()}

    pickle.dump(predicate_label, open(os.path.join(FLAGS.model_path, 'classes_to_labels.pkl'), "wb"), protocol=2)

    print(predicate_label)

    y = np.array([predicate_label[t[1]] for t in train_data])
    test_y = np.array([predicate_label[t[1]] for t in test_data])

    #y_test_predicted = test_df[classes_to_predict].values

    processed_train_comments = []
    for comment in raw_train_comments:
        processed_train_comments.append(preprocess_text(comment))

    processed_test_comments = []
    for comment in raw_test_comments:
        processed_test_comments.append(preprocess_text(comment))


    tokenizer = Tokenizer(num_words = MAX_NB_WORDS)
    tokenizer.fit_on_texts(processed_train_comments + processed_test_comments)

    pickle.dump(tokenizer, open(tokenizer_name, "wb"), protocol=2)

    train_sequences = tokenizer.texts_to_sequences(processed_train_comments)
    test_sequences = tokenizer.texts_to_sequences(processed_test_comments)

    print('found {} tokens in text.'.format(len(tokenizer.word_index)))

    train_data = pad_sequences(train_sequences, maxlen = MAX_SEQUENCE_LENGTH)

    final_test_data = pad_sequences(test_sequences, maxlen = MAX_SEQUENCE_LENGTH)

    print('train shape: {}'.format(train_data.shape))
    print("final_test_data.shape: {}".format(final_test_data.shape))
    print('shape of label(y) is {}'.format(y.shape))

    ##################################################
    ## preparing word embeddings.
    ##################################################

    print('preparing embedding matrix')
    word_index = tokenizer.word_index
    nb_words  = min(MAX_NB_WORDS, len(word_index)) + 1
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if(i> MAX_NB_WORDS):
            continue
        embedding_vector = embedding_index.get(word)
        if(embedding_vector is not None):
            embedding_matrix[i] = embedding_vector
    print('embedding matrix preparation complete')


    ##################################################
    ## train and validation split.
    ##################################################

    print('creating train and validation data by dividing train_data in 80:20 ratio')
    permutation = np.random.permutation(len(train_data))
    index_train = permutation[:int(len(train_data)*0.8)]
    index_validation = permutation[int(len(train_data)*0.2):]

    final_train_data = train_data[index_train]
    labels_of_train_data = y[index_train]

    final_validation_data = train_data[index_validation]
    labels_of_validation_data = y[index_validation]

    print('train data shape:', final_train_data.shape)
    print('validation data shape:', final_validation_data.shape)
    print('train and validation data are ready!!')

    ############################
    ## Keras model structure.
    ############################

    print("nb_words, EMBEDDING_DIM: {}".format([nb_words, EMBEDDING_DIM])) # [100000, 200]

    pickle.dump(embedding_matrix, open(os.path.join(FLAGS.model_path, 'embedding_matrix.pkl'), "wb"), protocol=2)

    model = make_model(nb_words, EMBEDDING_DIM, embedding_matrix, len(classes_to_predict))
    print(model.summary())

    # stamp = 'sentiment_with_lstm_and_glove_%.2f_%.2f'%(lstm_dropout_rate,dense_dropout_rate)
    # print(stamp)
    # best_model_path = stamp + '.h5'

    # best_model_path = os.path.join(FLAGS.model_path, 'checkpoint-{epoch:02d}-{val_loss:.2f}-{val_acc:.3f}.hdf5')
    best_model_path = os.path.join(FLAGS.model_path, CHECKPOINT_FILE)

    early_stopping = EarlyStopping(patience = 2)
    model_checkpoint = ModelCheckpoint(best_model_path, save_best_only = True, save_weights_only = False)

    tb = TensorBoard(log_dir=FLAGS.log_path,  # log 目录
                     histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
                     batch_size=32,  # 用多大量的数据计算直方图
                     write_graph=True,  # 是否存储网络结构图
                     write_grads=True,  # 是否可视化梯度直方图
                     write_images=True,  # 是否可视化参数
                     embeddings_freq=0,
                     embeddings_layer_names=None,
                     embeddings_metadata=None)

    hist = model.fit(x = final_train_data, y = labels_of_train_data,\
                     validation_data = (final_validation_data, labels_of_validation_data), \
                     epochs = 10, batch_size = 32, shuffle = True, \
                     callbacks = [early_stopping, model_checkpoint, tb])
    best_score = min(hist.history['val_loss'])
    print('best_score', best_score)

    #######################################
    ## time to make prediction!!!
    ########################################
    # y_test_predicted = model.predict([final_test_data], batch_size = 32, verbose = 1)

    print('模型评估')
    list_of_metrics = model.evaluate(x=final_test_data, y=test_y, batch_size=32)

    for index, metric in enumerate(model.metrics_names):
        print(metric + ':', str(list_of_metrics[index]))

def main(_):

    if FLAGS.train:
        train()
    else:
        evaluate()


if __name__ == "__main__":
    tf.app.run(main)





