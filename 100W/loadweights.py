"""
使用CNN进行文本分类

model = torch.load('./backup/szzxcorpus_trans.csv')
"""
"""
使用CNN进行文本分类
"""

import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
# from torch.autograd import Variable

from models import MultiCNNTextBNDeep


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# import pymysql
# from sqlalchemy import create_engine
# conn = pymysql.connect(host='localhost', user='root', password='root', db='akeec', charset='utf8')
# df1 = pd.read_sql('select words,lx from cnndataforpytorch', conn)
# x = df1.head()
# print(x)

###############################数据预处理#######################################

corpus = pd.read_csv('./corpusDataset/szzxcorpus_trans.csv', encoding='utf8')
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
with open('./utils/word2idx.pkl', 'rb') as f:
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
trainX = torch.LongTensor(np_wordnum2).to(device)
trainY = torch.LongTensor(np_label).to(device)




###############################数据预处理END####################################


n_word = len(word2idx)   # 不重复单词有多少个
context_size = 400       # 语句最大长度 文章长度 word为120 char为250
embedding_dim = 256      # embedding维度
content_dim = 200        # 文本的卷积核数
num_classes = 372        # 372个分类  t >= 0 && t < n_classes label need start at 0
linear_hidden_size = 373  # 全连接层隐藏元数目

BATCH_SIZE = 256  # 批大小
EPOCHS = 30  # 迭代次数


#model = MultiCNNTextBNDeep(vocab_size=n_word, embedding_dim=embedding_dim, content_dim=content_dim, linear_hidden_size=linear_hidden_size, num_classes=num_classes)
# fun_loss = nn.MultiLabelSoftMarginLoss() #这个报错

model = torch.load('./backup/cnntext_91_20180813_bak.pkl')

fun_loss = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model.train()
###################################模型end########################################



###################################模型end########################################

print(trainX.type())
# tensor 转换为dataset
# torch_dataset = TensorDataset(data_tensor=trainX, target_tensor=trainY)
torch_dataset = TensorDataset(trainX, trainY)  # 0.4 TensorDataset 参数为*tensors

# 填充dataloader
# 将torch_dataset置入Dataloader中
torch_loader = DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,  # 批大小
    # 若dataset中的样本数不能被batch_size整除的话，最后剩余多少就使用多少
    shuffle=True,   # 是否随机打乱顺序
    #num_workers=2,  # 多线程读取数据的线程数
    )



from tensorboardX import SummaryWriter
writer = SummaryWriter()

# for epoch in range(EPOCH):
for epoch in range(EPOCHS):
    print('Epoch:', epoch + 1, 'Training...')
    for step, (batch_x, batch_y) in enumerate(torch_loader):
        # 准备数据, 打包成Variable
        # b_x = Variable(batch_x)
        # b_y = Variable(batch_y)
        b_x = batch_x
        b_y = batch_y
        # 清空梯度缓存
        model.zero_grad()
        # 计算前向过程
        out = model(b_x).to(device)
        # 计算损失
        loss = fun_loss(out, b_y)
        #  反向传播, 优化一步
        loss.backward()
        optimizer.step()


        running_loss = 0.0
        running_acc = 0.0
        label = b_y
        #running_loss += loss.data[0] * label.size(0) #invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
        running_loss += loss.data.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()  # 本次batch中正确的个数总和
        accuracy = (pred == label).float().mean()  # 正确的平均数 平均真确多少个数
        running_acc += num_correct.data.item()  # 叠加运行的正确总数

        # (1) Log the scalar values
        info = {'loss': loss.data.item(), 'accuracy': accuracy.data.item(), 'step': (step+1)*BATCH_SIZE}
        print(info)

        ##################################### tensorboard
        comlums = (step+1)*BATCH_SIZE
        writer.add_scalar('data/loss', loss.data.item(), comlums)
        writer.add_scalar('data/accuracy', accuracy.data.item(), comlums)
        writer.add_text('zz/text', 'zz: this is epoch ' + str(comlums), comlums)
        features = batch_x
        lbael = batch_y
        writer.add_embedding(features, metadata=label)
        ##################################################
    if epoch%10==0:
        # 保存模型
        torch.save(model, f'./backup/cnntext_91_20180813_{epoch}.pkl')

writer.export_scalars_to_json("./test.json")
writer.close()