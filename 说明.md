# 中文句子分类模型--Bert

集成三类模型

​	Albert（[brightmart](https://github.com/brightmart)）

​	Bert（哈工大[Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)、哈工大[Chinese-ROBERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)、哈工大[MacBERT](https://github.com/ymcui/MacBERT)）

​	Electra（哈工大[Chinese-ELECTRA](https://github.com/ymcui/Chinese-ELECTRA)）



代码来自https://github.com/brightmart/albert_zh



**部分修改**

- 添加孪生网络，将输入从512（max）变为1024
- 合并模型训练与验证（train and dev）
- 尝试多种Bert模型的最终Linear层输出，参考Text-RCNN（max-pooling、conv2d）

