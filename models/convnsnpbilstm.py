
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class ConvNSNPBiLSTM(nn.Module):
    def __init__(self):
        super(ConvNSNPBiLSTM, self).__init__()
        self.num_inputs = 2
        self.outputs = 1
        self.num_channels = [64, 128, 256]
        self.kernel_size = 3
        self.dropout = 0.2
        self.pre_len = 2
        self.n_layers = 1
        self.relu = nn.ReLU()

        layers = []
        num_levels = len(self.num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = self.num_inputs if i == 0 else self.num_channels[i - 1]
            out_channels = self.num_channels[i]
            layers += [ConvModel(in_channels, out_channels, self.kernel_size, stride=1, dilation=dilation_size,
                                     padding=(self.kernel_size - 1) * dilation_size, dropout=self.dropout)]
        self.network = nn.Sequential(*layers)

        self.lstm = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.network(x)
        x = x.permute(0, 2, 1)

        output, (hidden, cell) = self.lstm(x)
        output = output.contiguous().view(32, 288, 2, 128)
        output = torch.mean(output, dim=2)
        y_pred = self.fc(output)
        y_pred = y_pred[:, -2:]
        return y_pred


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class ConvModel(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(ConvModel, self).__init__()
        self.conv = ConvNSNP1D(n_inputs, n_outputs)
        self.relu = nn.ReLU()

        self.conv1 = DilatedConvNSNP(n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout)
        self.conv2 = DilatedConvNSNP(n_outputs, n_outputs, kernel_size, stride, dilation, padding, dropout)
        self.conv3 = DilatedConvNSNP(n_outputs, n_outputs, kernel_size, stride, dilation, padding, dropout)

        self.net = nn.Sequential(self.conv1, self.conv2, self.conv3)

    def forward(self, x):
        out1 = self.conv(x)
        out2 = self.net(x)
        out = self.relu(out1 + out2)
        return out


class ConvNSNP1D(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(ConvNSNP1D, self).__init__()
        self.relu = nn.ReLU()
        self.Conv1d = nn.Conv1d(n_inputs, n_outputs, 1)
        self.Conv1d.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.relu(x)
        out = self.Conv1d(out)
        return out


class DilatedConvNSNP(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(DilatedConvNSNP, self).__init__()
        self.relu = nn.ReLU()
        self.conv = weight_norm(
        nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.dropout = nn.Dropout(dropout)
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv(out)
        out = self.chomp1(out)
        out = self.dropout(out)
        return out
