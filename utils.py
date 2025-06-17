import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from ptflops import get_model_complexity_info

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return torch.Tensor(sequence), torch.Tensor(label)


def create_dataloader(args, device):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>Create DataLoader" + args.data_path + "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    df = pd.read_csv(args.data_path)
    pre_len = args.pre_len
    train_window = args.window_size

    # move the target to the last column
    target_data = df[args.target]
    df = df.drop(args.target, axis=1)
    df = pd.concat((df, target_data), axis=1)

    cols_data = df.columns[2:]
    df_data = df[cols_data]

    true_data = df_data.values

    train_data = true_data[0:int(0.8 * len(true_data))]
    test_data = true_data[int(0.8 * len(true_data)): len(true_data)]
    print("Train size:", len(train_data), "Test size:", len(test_data))

    train_data_normalized = torch.FloatTensor(train_data).to(device)
    test_data_normalized = torch.FloatTensor(test_data).to(device)

    # use sliding window to get the data according to the  window size
    train_inout_seq = create_inout_sequences(train_data_normalized, train_window, pre_len, args)
    test_inout_seq = create_inout_sequences(test_data_normalized, train_window, pre_len, args)

    # Creat DataLoader
    train_dataset = TimeSeriesDataset(train_inout_seq)
    test_dataset = TimeSeriesDataset(test_inout_seq)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    print("The training set data is shared through the sliding window", len(train_inout_seq), "Convert to batch data:", len(train_loader))
    print("The testing set data is shared through the sliding window：", len(test_inout_seq), "Convert to batch data:", len(test_loader))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>" + args.data_path + "The creation of the DataLoader is completed<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    return train_loader, test_loader


def create_inout_sequences(input_data, tw, pre_len, args):
    inout_seq = []
    L = len(input_data)
    length = L - tw
    for i in range(0, length, 2):
        train_seq = input_data[i:i + tw]
        if (i + tw + pre_len) > len(input_data):
            break
        if args.feature == 'MS':
            train_label = input_data[:, 1:][i + tw:i + tw + pre_len]
        else:
            train_label = input_data[i + tw:i + tw + pre_len]
        inout_seq.append((train_seq, train_label))
    return inout_seq


def calculate_mae(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    return mae


def calculate_rmse(y_true, y_pred):
    rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
    return rmse


def calculate_mape(y_pred, y_true):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape


def calculate_cv(data_path, target):
    df = pd.read_csv(data_path)
    target_data = df[target]
    mean = np.mean(df[target])
    std = np.std(target_data)
    cv = std / mean
    print(cv)


class EarlyStopping:
    def __init__(self, patience, delta):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

def compute_model(model):
    input_res = (288, 2)
    macs, params = get_model_complexity_info(model, input_res, as_strings=True, print_per_layer_stat=True)
    print(f"model FLOPs: {macs}")
    print(f"model params: {params}")