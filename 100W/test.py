"""
使用CNN进行文本分类
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

corpus = pd.read_csv('./corpusDataset/szzx_test91.csv', encoding='utf8')
# print(corpus.head())
df2 = corpus[['WORDS', 'LX_MAP']].sample(frac=1)  # .head(20000)  # shuffle
# print(df2.head())
df2['WORDS'] = df2['WORDS'].apply(lambda x: x.split(' '))
# print(df2.head())

# np1 = np.array(df2)
# print(np1)

# allWords = []
# for x in df2['WORDS']:
#     allWords.extend(x)
#
# # print(len(allWords))  # 30000000
# setWords = set(allWords)  # 205581
# # print(len(setWords))  # 205581
#
# # word 转换为 number
# word2idx = {word: i for i, word in enumerate(setWords)}  # {'缺斤少两': 0, '时间': 1, ...}

import pickle
with open('word2idx.pkl', 'rb') as f:
    word2idx = pickle.load(f)


# number 转换为 word
idx2word = {word2idx[word]: word for word in word2idx}  # {0: '蓝焰', 1: '点', 2: '方向'}

# 语句通过词转换为序列
df2['WORDNUM'] = df2['WORDS'].apply(lambda x: [word2idx[m] for m in x])
# print(df2)

# 补全list 不够填充0

# for x in df2['WORDNUM']:
#     y = x+[0]*(400-len(x))
#     print(len(y))

df2['WORDNUM2'] = df2['WORDS'].apply(lambda x:[word2idx[m] for m in x]+[0]*(400-len(x)))
# df2['WORDNUM2'] = df2['WORDS'].apply(lambda x:len(x))

# list 转换为矩阵
np_wordnum2 = np.array(df2['WORDNUM2'].tolist())
np_label = np.array(df2['LX_MAP'].tolist())
print(np_wordnum2)

# narray转换为tensor
trainX = torch.LongTensor(np_wordnum2)
trainY = torch.LongTensor(np_label)


###############################数据预处理END####################################


###################################模型########################################

kernel_sizes = [1, 2, 3, 4]

# from keras import sequence
# sequence.pad_sequences

class MultiCNNTextBNDeep(nn.Module):
    def __init__(self, vocab_size, embedding_dim, content_dim, linear_hidden_size, num_classes):
        super(MultiCNNTextBNDeep, self).__init__()
        self.model_name = 'MultiCNNTextBNDeep'
        self.encoder = nn.Embedding(vocab_size, embedding_dim)

        content_convs = [nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=content_dim,
                      kernel_size=kernel_size),
            nn.BatchNorm1d(content_dim),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=content_dim,
                      out_channels=content_dim,
                      kernel_size=kernel_size),
            nn.BatchNorm1d(content_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=(context_size - kernel_size * 2 + 2))
        )
            for kernel_size in kernel_sizes]

        self.content_convs = nn.ModuleList(content_convs)

        self.fc = nn.Sequential(
            nn.Linear(len(kernel_sizes) * (content_dim), linear_hidden_size),
            nn.BatchNorm1d(linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(linear_hidden_size, num_classes)
        )

        # if opt.embedding_path:
        #     self.encoder.weight.data.copy_(t.from_numpy(np.load(opt.embedding_path)['vector']))

    def forward(self, content):
        content = self.encoder(content)

        content.detach()
        # if self.opt.static:
        #     content.detach()

        content_out = [content_conv(content.permute(0, 2, 1)) for content_conv in self.content_convs]
        conv_out = torch.cat((content_out), dim=1)
        reshaped = conv_out.view(conv_out.size(0), -1)
        logits = self.fc((reshaped))
        return logits


n_word = len(word2idx)   # 不重复单词有多少个
context_size = 400       # 语句最大长度 文章长度 400
embedding_dim = 256      # embedding维度
content_dim = 200        # 文本的卷积核数
num_classes = 372        # 372个分类
linear_hidden_size = 373  # 全连接层隐藏元数目

BATCH_SIZE = 128  # 批大小
EPOCH = 12  # 迭代次数


#model = MultiCNNTextBNDeep(vocab_size=n_word, embedding_dim=embedding_dim, content_dim=content_dim, linear_hidden_size=linear_hidden_size, num_classes=num_classes)

model = torch.load('cnntext_91_2.pkl')

# fun_loss = nn.MultiLabelSoftMarginLoss() #这个报错
fun_loss = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-3)

###################################模型end########################################



###################################模型end########################################

# tensor 转换为dataset
torch_dataset = TensorDataset(data_tensor=trainX, target_tensor=trainY)

# 填充dataloader
# 将torch_dataset置入Dataloader中
torch_loader = DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,  # 批大小
    # 若dataset中的样本数不能被batch_size整除的话，最后剩余多少就使用多少
    shuffle=True,   # 是否随机打乱顺序
    #num_workers=2,  # 多线程读取数据的线程数
    )



# for epoch in range(EPOCH):
for epoch in range(10):
    print('Epoch:', epoch + 1, 'Training...')
    for step, (batch_x, batch_y) in enumerate(torch_loader):
        # 准备数据, 打包成Variable
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)
        # 清空梯度缓存
        model.zero_grad()
        # 计算前向过程
        out = model(b_x)
        # 计算损失
        loss = fun_loss(out, b_y)
        #  反向传播, 优化一步
        loss.backward()
        optimizer.step()


        running_loss = 0.0
        running_acc = 0.0
        label = b_y
        running_loss += loss.data[0] * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()  # 本次batch中正确的个数总和
        accuracy = (pred == label).float().mean()  # 正确的平均数 平均真确多少个数
        running_acc += num_correct.data[0]  # 叠加运行的正确总数

        # (1) Log the scalar values
        info = {'loss': loss.data[0], 'accuracy': accuracy.data[0]}
        print(info)

