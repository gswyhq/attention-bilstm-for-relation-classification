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
from config import classes_to_predict

from config import MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, VALIDATION_SPLIT, num_lstm, num_dense, lstm_dropout_rate, dense_dropout_rate

from config import classes_to_predict, DEV_FILE, TRAIN_FILE, EMB_FILE, MODEL_PATH, LABELS_FILE, CHECKPOINT_FILE, EMBEDDING_MATRIX_FILE, TOKENIZER_FILE

def evaluate():

    checkpoint_file = os.path.join(MODEL_PATH, CHECKPOINT_FILE)
    classes_to_labels_flie = os.path.join(MODEL_PATH, LABELS_FILE)
    embedding_matrix_file = os.path.join(MODEL_PATH, EMBEDDING_MATRIX_FILE)
    # model_file = os.path.join(MODEL_PATH, 'model.pkl')
    tokenizer_file = os.path.join(MODEL_PATH, TOKENIZER_FILE)

    predicate_label = pickle.load(open(classes_to_labels_flie, 'rb'), encoding="iso-8859-1")
    embedding_matrix = pickle.load(open(embedding_matrix_file, 'rb'), encoding="iso-8859-1")
    # model = pickle.load(open(model_file, 'rb'), encoding="iso-8859-1")

    nb_words, EMBEDDING_DIM = [100000, 200]
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

    final_test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    print('模型评估')
    list_of_metrics = model.evaluate(x=final_test_data, y=test_y, batch_size=32)

    for index, metric in enumerate(model.metrics_names):
        print(metric + ':', str(list_of_metrics[index]))

def main():

    evaluate()


if __name__ == '__main__':
    main()