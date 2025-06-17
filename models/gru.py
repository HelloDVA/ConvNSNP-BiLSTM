import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Gru(nn.Module):
    def __init__(self):
        super(Gru, self).__init__()
        self.gru = nn.GRU(2, 128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        output, hidden = self.gru(x)
        y_pred = self.fc(output)
        y_pred = y_pred[:, -2:]
        return y_pred