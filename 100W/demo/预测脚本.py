"""

"""
import re
import jieba
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable

r = '[’!，。\n\s"#$%&\'()（）【】“”*+,-./:;<=>?@[\\]^_`{|}~]+'

def fc(line):
    """
    分词函数+去标点符号
    :param line:
    :return:
    """
    line = line.strip()
    line = re.sub(r, '', line)
    seg_list = jieba.cut(line, cut_all=False)  # 返回生成器
    return seg_list

###############################数据预处理#######################################



# mystr = """我 在 武昌区 地铁 4 号线 复兴路 站 里面 的 自助 图书馆 借 了 书 现在 还 不 进去 维修 人员 让 我 把 书 给 地铁站 的 工作人员 但 地铁 工作人员 不 给 我 收条 也 不留 联系电话 希望 政府 协调 处理 已转 地铁 集团 83481000 朱 女士 受理 并 要求 回复"""

mystr= "sdfsdf"

content_list = fc(mystr)


import pickle
with open('word2idx.pkl', 'rb') as f:
    word2idx = pickle.load(f)


# number 转换为 word
idx2word = {word2idx[word]: word for word in word2idx}  # {0: '蓝焰', 1: '点', 2: '方向'}

# 语句通过词转换为序列
# 补全list 不够填充0

wordnum2 = list(map(lambda x: word2idx[x], content_list))+[0]*(400-len(content_list))

# list 转换为矩阵
np_wordnum2 = np.array(wordnum2)

# narray转换为tensor
trainX = torch.LongTensor(np_wordnum2)
trainX = trainX.view(1, 400)  # 转换矩阵格式
#print(trainX.size())

###############################数据预处理END####################################


###################################模型########################################

kernel_sizes = [1, 2, 3, 4]

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




model = torch.load('cnntext_91_2.pkl')

# fun_loss = nn.MultiLabelSoftMarginLoss() #这个报错
fun_loss = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-3)

###################################模型end########################################


###################################预测########################################
model.eval()
b_x = Variable(trainX)
out = model(b_x)
_, pred = torch.max(out, 1)  # pred 为预测结果

print(_)
print(pred)

###################################预测end########################################
