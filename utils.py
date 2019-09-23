#!/usr/bin/python3
# coding: utf-8

import re


from keras.preprocessing.sequence import pad_sequences

def read_data(input_file):
    with open(input_file)as f:
        datas = f.readlines()
        datas = [t.strip().split('\t') for t in datas]
        datas = [t for t in datas if len(t) == 2]
    return datas



########################################
## Basic preprocessing of text data.
########################################

print('performing some basic preprocessing on data')

#regex for removing non-alphanumeric characters and spaces
remove_special_char = re.compile('r[^a-z\d]',re.IGNORECASE)

#regex to replace all numerics
replace_numerics = re.compile(r'\d+',re.IGNORECASE)

def preprocess_text(text, remove_stopwords=True, perform_stemming=True):
    """

    :param text:
    :param remove_stopwords:
    :param perform_stemming:
    :return:
    """
    text = text.lower()

    text = list(text)

    if (remove_stopwords):
        # stop_words = set(stopwords.words('english'))
        stop_words = set()
        text = [word for word in text if word not in stop_words]

    text = ' '.join(text)

    text = remove_special_char.sub('', text)
    text = replace_numerics.sub('n', text)

    if (perform_stemming):
        text = text.split()
        # stemmer = SnowballStemmer('english')
        # stemmed_words = [stemmer.stem(word) for word in text]
        stemmed_words = [w for w in text]
        text = ' '.join(stemmed_words)

    return text

def main():
    pass


if __name__ == '__main__':
    main()