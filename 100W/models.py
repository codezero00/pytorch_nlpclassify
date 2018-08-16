import torch
from torch import nn, optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###################################模型########################################

kernel_sizes = [1, 2, 3, 4]
context_size = 400
# from keras import sequence
# sequence.pad_sequences

class MultiCNNTextBNDeep(nn.Module):
    def __init__(self, vocab_size, embedding_dim, content_dim, linear_hidden_size, num_classes):
        super(MultiCNNTextBNDeep, self).__init__()
        self.model_name = 'MultiCNNTextBNDeep'
        self.encoder = nn.Embedding(vocab_size, embedding_dim).to(device) # cuda

        content_convs = [nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=content_dim,
                      kernel_size=kernel_size).to(device),
            nn.BatchNorm1d(content_dim).to(device),
            nn.ReLU(inplace=True).to(device),

            nn.Conv1d(in_channels=content_dim,
                      out_channels=content_dim,
                      kernel_size=kernel_size).to(device),
            nn.BatchNorm1d(content_dim).to(device),
            nn.ReLU(inplace=True).to(device),
            nn.MaxPool1d(kernel_size=(context_size - kernel_size * 2 + 2)).to(device)
        ).to(device)
            for kernel_size in kernel_sizes]

        self.content_convs = nn.ModuleList(content_convs).to(device)

        self.fc = nn.Sequential(
            nn.Linear(len(kernel_sizes) * (content_dim), linear_hidden_size).to(device),
            nn.BatchNorm1d(linear_hidden_size).to(device),
            nn.ReLU(inplace=True).to(device),
            nn.Linear(linear_hidden_size, num_classes).to(device)
        ).to(device)

        # if opt.embedding_path:
        #     self.encoder.weight.data.copy_(t.from_numpy(np.load(opt.embedding_path)['vector']))

    def forward(self, content):
        content = self.encoder(content)
        print(content.type())

        content.detach()
        # if self.opt.static:
        #     content.detach()

        content_out = [content_conv(content.permute(0, 2, 1)) for content_conv in self.content_convs]
        conv_out = torch.cat((content_out), dim=1)
        reshaped = conv_out.view(conv_out.size(0), -1)
        logits = self.fc((reshaped))
        return logits