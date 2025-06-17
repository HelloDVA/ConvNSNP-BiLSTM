import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class TCN(nn.Module):
    def __init__(self):
        super(TCN, self).__init__()
        # TCN
        self.num_inputs = 2   # 输入时间序列特征数
        self.outputs = 1       # 预测时间序列的特征数
        self.num_channels = [64, 128, 256]   # TCN 每层的情况
        self.kernel_size = 2     # 卷积核的个数
        self.dropout = 0.2
        # 构造每层的时域卷积 并存放起来
        layers = []
        num_levels = len(self.num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = self.num_inputs if i == 0 else self.num_channels[i - 1]
            out_channels = self.num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, self.kernel_size, stride=1, dilation=dilation_size,
                                     padding=(self.kernel_size - 1) * dilation_size, dropout=self.dropout)]
        self.network = nn.Sequential(*layers)

        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        # 经过TCN处理后的数据 shape(64, 288, 256)
        # permute转置，函数参数为原来数据的位置
        x = x.permute(0, 2, 1)
        x = self.network(x)
        x = x.permute(0, 2, 1)
        y_pred = self.fc(x)
        y_pred = y_pred[:, -2:]
        return y_pred


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.conv3 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp3 = Chomp1d(padding)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.chomp1, self.dropout1,
                                 self.conv2, self.relu2, self.chomp2, self.dropout2,
                                 self.conv3, self.relu3, self.chomp3, self.dropout3)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()