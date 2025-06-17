import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv1d(2, 128, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=1, stride=1)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=1, stride=1)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(0, 2, 1)

        y_pred = self.fc(x)
        y_pred = y_pred[:, -2:]
        return y_pred
