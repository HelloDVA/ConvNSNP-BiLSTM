import argparse
import math
import time
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from LSTNet import Model
import numpy as np
from optim import Optim
from utils import create_dataloader
from utils import compute_model


def calculate_mae(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    return mae


def calculate_rmse(y_true, y_pred):
    rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
    return rmse


def calculate_mape(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred)/y_true)) * 100
    return mape


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
    # data
    parser.add_argument('-data_path', type=str, default='../../data/gc11.csv', help="data path")
    parser.add_argument('-window_size', type=int, default=288, help="window size, window_size > pre_len")
    parser.add_argument('-pre_len', type=int, default=2, help="prediction size")
    parser.add_argument('-feature', type=str, default='MS', help='[M, S, MS],M->M,S->S,MS->S')
    parser.add_argument('-target', type=str, default='avgcpu')
    parser.add_argument('-input_size', type=int, default=2)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--gpu', type=int, default=None)
    # model
    parser.add_argument('--hidCNN', type=int, default=100, help='number of CNN hidden units')
    parser.add_argument('--hidRNN', type=int, default=100, help='number of RNN hidden units')
    parser.add_argument('--CNN_kernel', type=int, default=6, help='the kernel size of the CNN layers')
    parser.add_argument('--highway_window', type=int, default=24, help='The window size of the highway component')
    parser.add_argument('--clip', type=float, default=10., help='gradient clipping')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=54321, help='random seed')
    parser.add_argument('--log_interval', type=int, default=2000, metavar='N', help='report interval')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--skip', type=int, default=24)
    parser.add_argument('--hidSkip', type=int, default=5)
    parser.add_argument('--model', type=str, default='LSTNet', help='')
    parser.add_argument('--L1Loss', type=bool, default=True)
    parser.add_argument('--normalize', type=int, default=2)
    parser.add_argument('--output_fun', type=str, default='sigmoid')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit')
    args = parser.parse_args()

    args.cuda = args.gpu is not None
    if args.cuda:
        torch.cuda.set_device(args.gpu)

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    train_loader, test_loader = create_dataloader(args, args.device)

    model = Model(args)

    # use ptflops to compute the MAC and parameters of the model
    compute_model(model)

    if args.cuda:
        model.cuda()

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)

    if args.L1Loss:
        criterion = nn.L1Loss(size_average=False)
    else:
        criterion = nn.MSELoss(size_average=False)

    evaluateL2 = nn.MSELoss(size_average=False)
    evaluateL1 = nn.L1Loss(size_average=False)

    if args.cuda:
        criterion = criterion.cuda()
        evaluateL1 = evaluateL1.cuda()
        evaluateL2 = evaluateL2.cuda()

    optim = Optim(model.parameters(), args.optim, args.lr, args.clip)

    # train At any point you can hit Ctrl + C to break out of training early.
    try:
        print('begin training')
        BatchNums = math.ceil(len(train_loader.dataset) / train_loader.batch_size)
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            for i, data in enumerate(train_loader):
                seq, label = data
                seq, label = seq.to(args.device), label.to(args.device)
                model.zero_grad()
                output = model(seq)
                loss = criterion(output, label)
                loss.backward()
                grad_norm = optim.step()
                if i % 1 == 0:
                    print('Epoch[{}/{}], \t Batch[{}/{}], \t Loss:{:.6f}'.format(
                        epoch + 1, args.epochs, i + 1, BatchNums, loss.item()))
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # test
    predict = None
    test = None
    for data in test_loader:
        seq, label = data
        seq, label = seq.to(args.device), label.to(args.device)

        model.eval()
        output = model(seq)

        if predict is None:
            predict = output
            test = label
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, label))

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()

    mae = calculate_mae(Ytest, predict)
    rmse = calculate_rmse(Ytest, predict)
    mape = calculate_mape(Ytest, predict)

    print('rmse', rmse)
    print('mae', mae)
    print('mape', mape)

    plt.figure(figsize=(30, 8))
    plt.plot(predict, color='red')
    plt.plot(Ytest, color='blue')
    plt.grid(True)
    plt.show()




