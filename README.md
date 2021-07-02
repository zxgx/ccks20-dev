# Deprecated
**2021.7.3更新**

有人star了这个代码，感觉良心不安...更新一下

如果继续参加CCKS的KBQA任务的话，为了不浪费时间，不建议参考这个代码，我不会tensorflow，把源代码改成pytorch实现的过程很粗糙，最后拼在一起效果非常拉跨，建议直接参考获奖队伍源代码

# ccks20-dev
CCKS2019年第四名的复现  
源码：[ccks2019-ckbqa-4th-codes](https://github.com/duterscmy/ccks2019-ckbqa-4th-codes)

主要用pytorch实现了基于bert的文本匹配和序列标注，以及将知识图谱的检索由本地运行neo4j知识库改为了[gStore提供的http api](https://github.com/pkumod/gStore/blob/master/docs/API.md)  
另外有一些小的改动，没有实现原代码中的语义解析模块，不保证效果一致  
数据是用的2020年CCKS问答任务的训练集和测试集  
一些模块的总结概述和复现效果在pdf文件中

这个项目最大的价值应该是实现了一个完整的KBQA系统，包含了mention抽取和筛选，实体链接和筛选，候选路径的检索和筛选，并最终得到结果，不过各个模块细节还比较粗糙

## 运行
### 环境
1. 分别解压`corpus/data.zip`和`src/data/data.zip`到各自所在的目录下
2. 将主办方提供的PKUBASE的`pkubase-complete2.txt`和`pkubase-mention2ent.txt`解压到`PKUBASE`目录下
3. pytorch 1.4.0
4. transformers
5. jieba

其他的缺啥pip啥

### 训练
```
bash train.sh
```
shell文件我忘了怎么写，反正运行顺序是这样的，数据这些配置没问题的话直接运行就可以了  
**注意：** 训练大约花费19.5个小时，效率非常低，这些模块组合起来，效果也不算好

### 测试
```
cd src
python answer_bot.py
```
该脚本中，`valid`和`test`函数分别用于在验证集上计算系统的效果，和获取测试集上的结果  
**注意：** valid耗时大约2小时，test耗时大约4个多小时
