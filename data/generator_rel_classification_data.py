#!/usr/bin/python3
# coding: utf-8

import re
import json
import random
from pypinyin import pinyin, lazy_pinyin

# https://github.com/melancholicwang/lic2019-information-extraction-baseline
TRAIN_DATA_FILE = '/home/gswyhq/github_projects/lic2019-information-extraction-baseline/data/train_data.json'
DEV_DATA_FILE = '/home/gswyhq/github_projects/lic2019-information-extraction-baseline/data/dev_data.json'
TEST_DATA_FILE = '/home/gswyhq/github_projects/lic2019-information-extraction-baseline/data/test_demo.json'

SAVE_TRAIN_DATA_FILE = './data/train.txt'
SAVE_TEST_DATA_FILE = './data/test.txt'
SAVE_DEV_DATA_FILE = './data/dev.txt'

def get_pinyin(text, upper=False):
    """
    返回文本的拼音，若upper为真则所有字母都大写，否则仅仅首字母大写
    :param text: 文本
    :return: Shuxing
    """

    if isinstance(text, str):
        if upper:
            return ''.join(lazy_pinyin(text)).upper()
        else:
            return ''.join(lazy_pinyin(text)).capitalize()
    return ''

# BIOES   (B-begin，I-inside，O-outside，E-end，S-single)
#
# B 表示开始，I表示内部， O表示非实体 ，E实体尾部，S表示改词本身就是一个实体。

# BIO
# B 表示开始，I表示内部， O表示非实体
predicate_dict = {'祖籍': ['是什么', '在哪里'],
 '父亲': ['是什么', '是谁'],
 '总部地点': ['是什么', '在哪里'],
 '出生地': ['是什么', '在哪里'],
 '目': ['是什么'],
 '面积': ['是什么', '有多大'],
 '简称': ['是什么'],
 '上映时间': ['是什么', '什么时候'],
 '妻子': ['是什么', '是谁'],
 '所属专辑': ['是什么', '是啥'],
 '注册资本': ['是什么', '有多少'],
 '首都': ['是什么', '在哪里'],
 '导演': ['是什么', '是谁'],
 '字': ['是什么'],
 '身高': ['是什么', '有多高'],
 '出品公司': ['是什么', '哪个公司'],
 '修业年限': ['是什么', '多久'],
 '出生日期': ['是什么', '什么时候'],
 '制片人': ['是什么', '是谁'],
 '母亲': ['是什么', '是谁'],
 '编剧': ['是什么'],
 '国籍': ['是什么', '哪个国家'],
 '海拔': ['是多少', '有多高'],
 '连载网站': ['是什么'],
 '丈夫': ['是什么', '是谁'],
 '朝代': ['是什么'],
 '民族': ['是什么'],
 '号': ['是什么'],
 '出版社': ['是什么'],
 '主持人': ['是什么', '是谁'],
 '专业代码': ['是什么'],
 '歌手': ['是什么', '是谁'],
 '作词': ['是什么', '是谁'],
 '主角': ['是什么', '是谁'],
 '董事长': ['是什么', '是谁'],
 '成立日期': ['是什么', '是什么时候'],
 '毕业院校': ['是什么', '哪个学校'],
 '占地面积': ['是什么', '有多大'],
 '官方语言': ['是什么'],
 '邮政编码': ['是什么'],
 '人口数量': ['是什么', '有多少'],
 '所在城市': ['是什么', '在哪里'],
 '作者': ['是什么', '是谁'],
 '作曲': ['是什么', '是谁'],
 '气候': ['是什么', '冷不冷'],
 '嘉宾': ['是什么', '是谁'],
 '主演': ['是什么', '是谁'],
 '改编自': ['是什么'],
 '创始人': ['是什么', '是谁']}

def format_tran(input_file, output_file, format='IOB'):
    with open(input_file) as f:
        line = f.readline()
        with open(output_file, 'a+', encoding='utf-8')as f2:
            while line:
                data = json.loads(line)
                postag = data.get('postag', [])
                text = data['text']
                spo_list = data.get('spo_list', [])
                if not spo_list or not postag:
                    line = f.readline()
                    continue
                text = text.replace('\t', '')
                entity_dict = {}
                for spo in spo_list:
                    predicate = spo["predicate"]
                    object_type = spo["object_type"]
                    subject_type = spo["subject_type"]
                    object = spo["object"]
                    subject = spo["subject"]
                    entity_dict.setdefault(subject.upper(), get_pinyin(subject_type, upper=True))
                    entity_dict.setdefault(object.upper(), get_pinyin(object_type, upper=True))

                    if len(text) <= 60:
                        f2.write("{}\t{}\n".format(text, predicate))

                    split_text = re.split('[,，。？；！]', text)
                    for sub_text in split_text:
                        if subject in sub_text and object in sub_text:
                            sub_text = sub_text.replace(object, random.choice(predicate_dict.get(predicate)))
                            f2.write("{}\t{}\n".format(sub_text, predicate))
                    
                    f2.write("{}的{}{}\t{}\n".format(subject, predicate, random.choice(predicate_dict.get(predicate)), predicate))

                # for word_pos in postag:
                #     word = word_pos.get('word').upper()
                #     word = word.strip()
                #     word_pinyin = entity_dict.get(word)
                #     if word_pinyin:
                #         f2.write('\n')

                line = f.readline()
            print('保存文件： {}'.format(output_file))


def main():
    for input_file, output_file in zip([TRAIN_DATA_FILE, DEV_DATA_FILE, TEST_DATA_FILE], [SAVE_TRAIN_DATA_FILE, SAVE_DEV_DATA_FILE, SAVE_TEST_DATA_FILE]):
        format_tran(input_file, output_file)


if __name__ == '__main__':
    main()