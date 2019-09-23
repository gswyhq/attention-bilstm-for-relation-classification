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
from config import classes_to_predict, MODEL_PATH, LABELS_FILE, CHECKPOINT_FILE, EMBEDDING_MATRIX_FILE, TOKENIZER_FILE, nb_words, EMBEDDING_DIM
from api.logger import logger


class RelClass():
    def __init__(self):
        checkpoint_file = os.path.join(MODEL_PATH, CHECKPOINT_FILE)
        classes_to_labels_flie = os.path.join(MODEL_PATH, LABELS_FILE)
        embedding_matrix_file = os.path.join(MODEL_PATH, EMBEDDING_MATRIX_FILE)
        # model_file = os.path.join(MODEL_PATH, 'model.pkl')
        tokenizer_file = os.path.join(MODEL_PATH, TOKENIZER_FILE)

        self.predicate_label = pickle.load(open(classes_to_labels_flie, 'rb'), encoding="iso-8859-1")
        embedding_matrix = pickle.load(open(embedding_matrix_file, 'rb'), encoding="iso-8859-1")
        # model = pickle.load(open(model_file, 'rb'), encoding="iso-8859-1")

        label2id = {k: t.argmax() for k, t in self.predicate_label.items()}
        self.id2label = {_id: label for label, _id in label2id.items()}

        # nb_words, EMBEDDING_DIM = [8179, 200] #[100000, 200]
        self.model = make_model(nb_words, EMBEDDING_DIM, embedding_matrix, classes_to_predict)
        self.model.load_weights(checkpoint_file)
        # model = load_model(checkpoint_file)

        self.tokenizer = pickle.load(open(tokenizer_file, 'rb'), encoding="iso-8859-1")

    def evaluate_line(self, line):
        if isinstance(line, str):
            raw_test_comments = [line]
        elif isinstance(line, (list, tuple)):
            raw_test_comments = [question for question, entity_dict in line]
        else:
            raise ValueError('【格式错误】question 字段值应该为字符串或列表！')
        processed_test_comments = []
        for comment in raw_test_comments:
            processed_test_comments.append(preprocess_text(comment))

        test_sequences = self.tokenizer.texts_to_sequences(processed_test_comments)

        final_test_data = pad_sequences(test_sequences, maxlen=150)

        rets = self.model.predict(x=final_test_data, batch_size=1)

        ret = []
        for pred, question in zip(rets, raw_test_comments):
            # argsort函数返回的是数组值从小到大的索引值
            sort_index = pred.argsort()
            pred_ret = [{'question': question, 'intent': self.id2label[_index], 'score': float(pred[_index])} for _index in sort_index[-5:][::-1]]
            ret.append(pred_ret)
            # label = self.id2label[pred.argmax()]
            # score = float(pred.max())
            # ret.append([{'question': question, 'intent': label, 'score': score}])

        logger.info("问句`{}`实体识别的结果：{}".format(line, ret))
        return ret



rel_class = RelClass()

def main():
    ret = rel_class.evaluate_line('珠穆朗玛峰有多高')
    print(ret)


if __name__ == "__main__":
    # tf.app.run(main)
    main()

# docker run -e 'POST={"question": "姚明有多高", "pid": "123456", "extract_tags": true}' -it --rm -t devth/alpine-bench -t 60 -c 30 http://192.168.3.164:8000/parser
