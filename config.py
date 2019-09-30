#!/usr/bin/python3
# coding: utf-8

MAX_SEQUENCE_LENGTH = 150
MAX_NB_WORDS = 100000
num_lstm = 100
VALIDATION_SPLIT = 0.1
lstm_dropout_rate = 0.25
dense_dropout_rate = 0.25
num_dense = 128
# EMBEDDING_DIM = 200  # 词向量的维数

# nb_words, EMBEDDING_DIM = [6865, 200] # [8179, 200] # [100000, 200]

DEV_FILE = './data/dev.txt'
TRAIN_FILE = './data/train.txt'
EMB_FILE = '../data/Tencent_AILab_ChineseEmbedding.txt'

MODEL_PATH = 'model' # "./model"
LOG_PATH = './log'

CHECKPOINT_FILE = 'model.hdf5'
# CHECKPOINT_FILE = 'checkpoint-02-0.39-0.830.hdf5' # 'checkpoint-02-0.03-0.986.hdf5' # 'checkpoint-01-0.04-0.985.hdf5' #'checkpoint-03-0.08-0.980.hdf5' # 'checkpoint-01-0.08-0.979.hdf5'
LABELS_FILE = 'classes_to_labels.pkl'
EMBEDDING_MATRIX_FILE = 'embedding_matrix.pkl'
TOKENIZER_FILE = 'tokenizer.pkl'

# classes_to_predict = ['祖籍', '父亲', '总部地点', '出生地', '目', '面积', '简称', '上映时间', '妻子', '所属专辑', '注册资本', '首都', '导演', '字', '身高', '出品公司', '修业年限', '出生日期', '制片人', '母亲', '编剧', '国籍', '海拔', '连载网站', '丈夫', '朝代', '民族', '号', '出版社', '主持人', '专业代码', '歌手', '作词', '主角', '董事长', '成立日期', '毕业院校', '占地面积', '官方语言', '邮政编码', '人口数量', '所在城市', '作者', '作曲', '气候', '嘉宾', '主演', '改编自', '创始人'] # '成立日期',
# classes_to_predict = ['占地面积', '连载网站', '预防措施', '编剧', '导演', '忌吃食物', '改编自', '主演', '定义', '概念', '字', '出版社', '作曲', '简称', '药品详情', '推荐药品', '出品公司', '气候', '官方语言', '邮政编码', '身高', '歌手', '并发疾病', '简介', '治疗费用', '治疗方式', '创始人', '嘉宾', '修业年限', '传染性', '主持人', '治疗科室', '易感人群', '治愈概率', '出生地', '目', '母亲', '总部地点', '所属专辑', '治疗周期', '诊断检查项目', '专业代码', '症状', '人口数量', '分类', '首都', '丈夫', '主角', '发病率', '宜吃食物', '民族', '制片人', '所在城市', '病因', '作者', '基本医疗保险', '毕业院校', '号', '常用药品', '妻子', '父亲', '祖籍', '面积', '作词', '成立日期', '出生日期', '注册资本', '国籍', '朝代', '推荐食谱', '董事长', '海拔', '上映时间']

def main():
    pass


if __name__ == '__main__':
    main()
