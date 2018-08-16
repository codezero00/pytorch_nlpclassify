"""
数据预处理，将 n_word 最大单词个数序列化然后存储
"""

import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.autograd import Variable

# import pymysql
# from sqlalchemy import create_engine
# conn = pymysql.connect(host='localhost', user='root', password='root', db='akeec', charset='utf8')
# df1 = pd.read_sql('select words,lx from cnndataforpytorch', conn)
# x = df1.head()
# print(x)

###############################数据预处理#######################################

corpus = pd.read_csv('../corpusDataset/szzxcorpus_trans.csv', encoding='utf8')
# print(corpus.head())
df2 = corpus[['WORDS', 'LX_MAP']].sample(frac=1)  # .head(20000)  # shuffle
# print(df2.head())
df2['WORDS'] = df2['WORDS'].apply(lambda x: x.split(' '))
# print(df2.head())

# np1 = np.array(df2)
# print(np1)

allWords = []
for x in df2['WORDS']:
    allWords.extend(x)

# print(len(allWords))  # 30000000
setWords = set(allWords)  # 205581
# print(len(setWords))  # 205581

# word 转换为 number
word2idx = {word: i for i, word in enumerate(setWords)}  # {'缺斤少两': 0, '时间': 1, ...}
import pickle
#pickle.dumps(d)
with open('word2idx.pkl', 'wb') as f:
    pickle.dump(word2idx, f)

# # number 转换为 word
# idx2word = {word2idx[word]: word for word in word2idx}  # {0: '蓝焰', 1: '点', 2: '方向'}

# # 语句通过词转换为序列
# df2['WORDNUM'] = df2['WORDS'].apply(lambda x: [word2idx[m] for m in x])
# # print(df2)
#
# # 补全list 不够填充0
#
# # for x in df2['WORDNUM']:
# #     y = x+[0]*(400-len(x))
# #     print(len(y))
#
# df2['WORDNUM2'] = df2['WORDS'].apply(lambda x:[word2idx[m] for m in x]+[0]*(400-len(x)))
# # df2['WORDNUM2'] = df2['WORDS'].apply(lambda x:len(x))
#
# # list 转换为矩阵
# np_wordnum2 = np.array(df2['WORDNUM2'].tolist())
# np_label = np.array(df2['LX_MAP'].tolist())
# print(np_wordnum2)
#
# # narray转换为tensor
# trainX = torch.LongTensor(np_wordnum2)
# trainY = torch.LongTensor(np_label)


###############################数据预处理END####################################

