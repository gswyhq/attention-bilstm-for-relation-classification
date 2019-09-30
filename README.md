
训练：
python3 main.py --train true

训练数据格式

`question\tlabels`
```shell
《报告老板之豪言壮旅》的主演是谁	主演
黄道周（1585-1646），是哪里人	出生地

```
原始训练数据基于[三元组文件](https://github.com/melancholicwang/lic2019-information-extraction-baseline/data)数据利用`./data/generator_rel_classification_data.py`生成

中文词向量文件[Tencent_AILab_ChineseEmbedding.txt](https://ai.tencent.com/ailab/nlp/data/Tencent_AILab_ChineseEmbedding.tar.gz)

curl http://localhost:8000/nlp/classifynet/infer -d '{"pid": "adbcd", "question":[["天龙八部是谁主演的", {}], ["太平洋的面积是多少", {}], ["百度总部在哪", {}], ["珠穆朗玛峰有多高", {}], ["百度公司的董事长是谁", {}], ["太平洋有多大", {}], ["如何治疗感冒", {}], ["乳腺癌的症状有哪些？", {}], ["最近老流鼻涕怎么办？", {}], ["为什么有的人会失眠？", {}], ["失眠有哪些并发症？", {}], ["失眠的人不要吃啥？", {}], ["耳鸣了吃点啥？", {}], ["哪些人最好不好吃蜂蜜？", {}], ["鹅肉有什么好处？", {}], ["肝病要吃啥药？", {}], ["板蓝根颗粒能治啥病？", {}], ["脑膜炎怎么才能查出来？", {}], ["怎样才能预防肾虚？", {}], ["感冒要多久才能好？", {}], ["高血压要怎么治？", {}], ["白血病能治好吗？", {}], ["什么人容易得高血压？", {}], ["糖尿病", {}], ["全血细胞计数能查出啥来", {}]]}'

{"msg": "请求成功", "code": 0, "data": [[{"question": "天龙八部是谁主演的", "intent": "主演", "score": 0.14983052015304565}], [{"question": "太平洋的面积是多少", "intent": "主演", "score": 0.14983052015304565}]]}



/home/gswyhq/github_projects/fastTextCLF

https://github.com/senofgithub/keras_text_classifier

https://github.com/LeBronBao/Text_Classifier

https://github.com/AlexGidiotis/Document-Classifier-LSTM

https://github.com/littleflow3r/attention-bilstm-for-relation-classification

