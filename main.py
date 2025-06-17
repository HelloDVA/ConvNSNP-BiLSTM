import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from models.convnsnpbilstm import ConvNSNPBiLSTM
from utils import create_dataloader, EarlyStopping
from utils import calculate_rmse, calculate_mae, calculate_mape


np.random.seed(0)


def getargs():
    parser = argparse.ArgumentParser(description='Time Series forecast')
    parser.add_argument('-data_path', type=str, default='./data', help="dataset path")
    parser.add_argument('-dataset', type=str, default='', help="dataset", required=True)
    parser.add_argument('-lr', type=float, default=0, help="learning rate", required=True)
    parser.add_argument('-model', type=str, default='T', help="model name")
    parser.add_argument('-window_size', type=int, default=288, help="time window size (window_size > pre_len)")
    parser.add_argument('-pre_len', type=int, default=2, help="prediction length")
    parser.add_argument('-feature', type=str, default='MS', help='[M, S, MS],M->M,S->S,M-S')
    parser.add_argument('-target', type=str, default='avgcpu')
    parser.add_argument('-input_size', type=int, default=2)
    parser.add_argument('-epochs', type=int, default=100, help="epoch")
    parser.add_argument('-batch_size', type=int, default=32, help="bacth size")
    args = parser.parse_args()
    if args.feature == 'MS' or args.feature == 'S':
        args.output_size = 1
    else:
        args.output_size = args.input_size
    args.data_path = args.data_path + '/' + args.dataset + '.csv'
    return args


def train_model(model, args):
    print(f">>>>>>>>>>>>>>>>>>>>>>>>>Start Train<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    epochs = args.epochs
    model.train()  # train mode

    results_loss = []
    early_stopping = EarlyStopping(patience=10, delta=0.0001)
    for i in tqdm(range(epochs)):
        loss = []
        for seq, labels in train_loader:
            optimizer.zero_grad()
            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()
            loss.append(single_loss.detach().cpu().numpy())
        tqdm.write(f"\t Epoch {i + 1} / {epochs}, Loss: {sum(loss) / len(loss)}")
        results_loss.append(sum(loss) / len(loss))
        early_stopping(sum(loss) / len(loss))
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {i}")
            break
        time.sleep(0.1)
    return model


def test_model(model, args):
    print(f">>>>>>>>>>>>>>>>>>>>>>>>>Start Test<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    model.eval()  # eval mode
    maes = []
    rmses = []
    mapes = []
    results = []
    labels = []

    for seq, label in test_loader:
        pred = model(seq)
        mae = calculate_mae(pred.detach().cpu().numpy(), np.array(label.detach().cpu()))
        rmse = calculate_rmse(pred.detach().cpu().numpy(), np.array(label.detach().cpu()))
        mape = calculate_mape(pred.detach().cpu().numpy(), np.array(label.detach().cpu()))
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        pred = pred.cpu().detach().numpy()
        label = label.cpu().numpy()
        pred = pred[:, :, 0]
        label = label[:, :, 0]
        # save result
        for i in range(len(pred)):
            for j in range(2):
                results.append(pred[i][j])
                labels.append(label[i][j])
    print("Test Dataset RMSE:", np.mean(rmses))
    print("Test Dataset MAE:", np.mean(maes))
    print("Test Dataset MAPE:", np.mean(mapes))
    plot_result(labels, results)


def plot_result(labels, results):
    plt.figure(figsize=(20, 6))
    plt.xlabel('Time/5min', fontsize=20)
    plt.ylabel('AvgCPU', fontsize=20)
    plt.title("Dataset name", fontsize=20)
    plt.plot(labels, label='TrueValue', color='black')
    plt.plot(results, label='Prediction', color='blue')
    plt.show()



if __name__ == '__main__':
    args = getargs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("The Device:", device)
    # make data loader
    train_loader, test_loader = create_dataloader(args, device)
    # make model
    print(f">>>>>>>>>>>>>>>>>>>>>>>>>Start construct{args.model}model<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    model = ConvNSNPBiLSTM().to(device)
    model = train_model(model, args)
    test_model(model, args)





















