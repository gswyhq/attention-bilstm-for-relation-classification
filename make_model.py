#!/usr/bin/python3
# coding: utf-8

import os
import pickle

from keras import Input, Model, initializers, backend as K
from keras.engine import Layer
from keras.layers import Embedding, LSTM, Dropout, Dense, BatchNormalization
from config import MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, VALIDATION_SPLIT, num_lstm, num_dense, lstm_dropout_rate, dense_dropout_rate

class Attention(Layer):
     # Input shape 3D tensor with shape: `(samples, steps, features)`.
     # Output shape 2D tensor with shape: `(samples, features)`.

    def __init__(self, step_dim,W_regulizer = None,b_regulizer = None,
                 W_constraint = None, b_constraint = None,bias = True,**kwargs):

        self.W_regulizer = W_regulizer
        self.b_regulizer = b_regulizer

        self.W_constraint = W_constraint
        self.b_constraint = b_constraint

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        self.init = initializers.get('glorot_uniform')
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='kernel',
                                      shape=(input_shape[-1],),
                                      initializer= self.init,
                                      constraint = self.W_constraint,
                                      regularizer = self.W_regulizer,
                                      # name = '{}_W'.format(self.name)
                                 )

        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regulizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        super(Attention, self).build(input_shape)


    def call(self, x, mask=None):

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:

            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a

        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

def make_model(nb_words, EMBEDDING_DIM, embedding_matrix, class_num):

    embedding_layer = Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    lstm_layer = LSTM(num_lstm, dropout=lstm_dropout_rate, recurrent_dropout=lstm_dropout_rate, return_sequences=True)

    input_comment = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequence = embedding_layer(input_comment)
    x = lstm_layer(embedded_sequence)
    x = Dropout(dense_dropout_rate)(x)
    merged = Attention(MAX_SEQUENCE_LENGTH)(x)
    merged = Dense(num_dense, activation='relu')(merged)
    merged = Dropout(dense_dropout_rate)(merged)
    merged = BatchNormalization()(merged)
    preds = Dense(class_num, activation='softmax')(merged) # sigmoid

    model = Model(inputs=[input_comment], outputs=preds)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # binary_crossentropy
    # print(model.summary())

    # pickle.dump(model, open(os.path.join(FLAGS.model_path, 'model.pkl'), "wb"), protocol=2)

    return model



def main():
    pass


if __name__ == '__main__':
    main()

