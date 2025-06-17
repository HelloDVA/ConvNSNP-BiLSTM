import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class CnnLstm(nn.Module):
    def __init__(self):
        super(CnnLstm, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=1, stride=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride=1)
        self.lstm = nn.LSTM(256, 128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(0, 2, 1)

        output, (hidden, cell) = self.lstm(x)
        y_pred = self.fc(output)
        y_pred = y_pred[:, -2:]
        return y_pred