import torch
from torch import nn

class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        #首先通过一维卷据,一维卷积的kernel_size = in_channels*kernel_size。但是它只在length上滑动
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=5,kernel_size=3,stride=1)  #out_channels词嵌入维度
        self.max_pool1 = nn.MaxPool1d(kernel_size=2,stride=2)  #尺寸减小一半
        self.conv2 = nn.Conv1d(in_channels=5,out_channels=10,kernel_size=4,stride=1)
        self.max_pool2 = nn.MaxPool1d(kernel_size=2,stride=2)
        self.conv3 = nn.Conv1d(in_channels=10, out_channels=20,kernel_size=4,stride=1)
        self.max_pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=20*35,out_features=30)
        self.fc2 = nn.Linear(in_features=30,out_features=20)
        self.fc3 = nn.Linear(in_features=20,out_features=5)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = x.permute(0,2,1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = self.max_pool3(x)  #batchsize,channel,seq_length
        x = x.contiguous().view(-1,20*35)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


