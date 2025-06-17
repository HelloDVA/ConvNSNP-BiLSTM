import torch
import numpy as np;
from torch.autograd import Variable


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, cuda, horizon, window):
        self.cuda = cuda
        self.P = window
        self.h = horizon

        fin = open(file_name)
        self.dat = np.loadtxt(fin, delimiter=',', skiprows=1, usecols=(2, 3))  # 从文件中读取数据，使用skiprows跳过无用的列

        # self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape

        self.scale = np.ones(self.m)
        self._split(int(train * self.n), int((1-train)*self.n))

        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)

        if self.cuda:
            self.scale = self.scale.cuda();
        self.scale = Variable(self.scale);

        self.rse = normal_std(tmp);
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)));


    def _split(self, train, test):

        train_set = range(self.P + self.h - 1, train)
        test_set = range(train, self.n)
        self.train = self._batchify(train_set)
        self.test = self._batchify(test_set)

    def _batchify(self, idx_set):

        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, 1))

        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :][0])

        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt];
            Y = targets[excerpt];
            if (self.cuda):
                X = X.cuda();
                Y = Y.cuda();
            yield Variable(X), Variable(Y);
            start_idx += batch_size