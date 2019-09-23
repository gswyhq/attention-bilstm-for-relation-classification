#!/usr/bin/python3
# coding: utf-8

import os
import pickle
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from utils import read_data, preprocess_text
from make_model import make_model
from config import classes_to_predict, MODEL_PATH, DEV_FILE, LABELS_FILE, CHECKPOINT_FILE, EMBEDDING_MATRIX_FILE, TOKENIZER_FILE, nb_words, EMBEDDING_DIM

def predict():

    checkpoint_file = os.path.join(MODEL_PATH, CHECKPOINT_FILE)
    classes_to_labels_flie = os.path.join(MODEL_PATH, LABELS_FILE)
    embedding_matrix_file = os.path.join(MODEL_PATH, EMBEDDING_MATRIX_FILE)
    # model_file = os.path.join(MODEL_PATH, 'model.pkl')
    tokenizer_file = os.path.join(MODEL_PATH, TOKENIZER_FILE)

    predicate_label = pickle.load(open(classes_to_labels_flie, 'rb'), encoding="iso-8859-1")
    embedding_matrix = pickle.load(open(embedding_matrix_file, 'rb'), encoding="iso-8859-1")
    # model = pickle.load(open(model_file, 'rb'), encoding="iso-8859-1")

    label2id = {k: t.argmax() for k, t in predicate_label.items()}
    id2label = {_id: label for label, _id in label2id.items()}

    model = make_model(nb_words, EMBEDDING_DIM, embedding_matrix, classes_to_predict)
    model.load_weights(checkpoint_file)
    # model = load_model(checkpoint_file)

    tokenizer = pickle.load(open(tokenizer_file, 'rb'), encoding="iso-8859-1")

    test_data = read_data(DEV_FILE)

    raw_test_comments = [t[0] for t in test_data]

    test_y = np.array([predicate_label[t[1]] for t in test_data])

    processed_test_comments = []
    for comment in raw_test_comments:
        processed_test_comments.append(preprocess_text(comment))

    test_sequences = tokenizer.texts_to_sequences(processed_test_comments)

    final_test_data = pad_sequences(test_sequences, maxlen=150)
    # print('test_data', test_data[:3])
    print('模型评估')
    ret = model.predict(x=final_test_data, batch_size=1)
    # print('预测结果：', ret)
    # print('标注', '预测', '问题')
    rets = []
    for label, pred, question in zip(test_y, ret, test_data):
        print(id2label[label.argmax()], id2label[pred.argmax()], question)
        rets.append([id2label[label.argmax()], id2label[pred.argmax()], question])

    print('正确率：{}'.format(len([t for t in rets if t[0]==t[1]])/len(rets)))

def main():
    predict()

if __name__ == '__main__':
    main()
