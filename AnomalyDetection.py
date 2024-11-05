import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.model_selection import KFold

from models.LSTMAE import LSTMAE
from models.LSTM_PRED import LSTMAEPRED
from models.LSTMAE_RECON import LSTMAE_RECON
from train_utils import train_model, eval_model

parser = argparse.ArgumentParser(description='S&P code')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=2500, metavar='N', help='number of epochs to train')
parser.add_argument('--optim', default='Adam', type=str, help='Optimizer to use')
parser.add_argument('--hidden-size', type=int, default=64, metavar='N', help='LSTM hidden state size')
parser.add_argument('--embedding-size', type=int, default=32, metavar='N', help='LSTM bottle neck layer size')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate')
parser.add_argument('--input-size', type=int, default=256, metavar='N', help='input size')
parser.add_argument('--dropout', type=float, default=0.0, metavar='D', help='dropout ratio')
parser.add_argument('--wd', type=float, default=0, metavar='WD', help='weight decay')
parser.add_argument('--grad-clipping', type=float, default=1, metavar='GC', help='gradient clipping value')
parser.add_argument('--log-interval', type=int, default=1, metavar='N', help='how many batch iteration to log status')
parser.add_argument('--model-dir', default='trained_models', help='directory of model for saving checkpoint')
parser.add_argument('--model-type', default='LSTMAE_RECON', help='type of model to use: LSTMAE, LSTMAE_PRED, or LSTMAE_RECON for '
                                                           'prediction')
parser.add_argument('--seq-len', default=64, help='sequence full length')
parser.add_argument('--features', default=1, help='number of features in input data')
parser.add_argument('--val-type', default='SquareWaveWithNoise', help='index for data column')
parser.add_argument('--cross-val', type=int, default=4, help='number of cross validation experiments to run')
parser.add_argument('--data-path', type=str, default=r'C:\Users\dwmccauley\OneDrive - BWX Technologies Inc\Documents\LSTMAE\LSTM_AutoEncoder\Data\square_wave_with_large_spikes0.csv',
                    help='Path to the S&P csv data')

args = parser.parse_args(args=[])

kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class square_dataset(torch.utils.data.Dataset):
    def __init__(self, df, vals_scaling_metadata):
        self.df = df
        self.vals_scaling_metadata = vals_scaling_metadata

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):        
        return torch.FloatTensor(self.df[index])


def main():
    best_model, best_loss, test_df, vals_scaling_metadata = run_experiment()

    # Generate final test iterator and plot reconstructed graphs
    test_iter = torch.utils.data.DataLoader(square_dataset(test_df, vals_scaling_metadata, args.seq_len),
                                            batch_size=args.batch_size, shuffle=False)
    plot_images(best_model, test_iter, num_to_plot=3)

    # Save model
    torch.save(best_model.state_dict(),
               os.path.join(args.model_dir, f'anomaly detection model={args.model_type}_hs={args.hidden_size}_bs={args.batch_size}'
                                            f'_epochs={args.epochs}_clip={args.grad_clipping}.pt'))


def rec_pred_loss(data_rec, data, out_preds, real_pred):
    """
    Function that calculated both reconstruction and prediction loss
    :param data_rec: reconstructed data instance
    :param data: original data instance
    :param out_preds: predicted next price
    :param real_pred: actual next price
    :return: the reconstruction loss and prediction loss
    """
    mse_rec = torch.nn.MSELoss(reduction='sum')(data_rec, data)
    mse_pred = torch.nn.MSELoss(reduction='sum')(out_preds, real_pred)

    return mse_rec, mse_pred


def plot_stock_values(df, ticker):
    """
    This function gets a dataFrame of stocks and their data, and a a ticker id, and plots the price graph (e.g., high)
    :param df: dataFrame containing stocks prices
    :param ticker: ticker id
    :return:
    """
    ticker_df = df[df.symbol == ticker]
    ticker_df.sort_values(by=['date'], inplace=True)

    ticker_df.plot(x='date', y=args.price_type, label=f'{ticker} {args.price_type} daily values',
                   ylabel=f'Stock {args.price_type} price')
    plt.show()


def run_experiment():
    """
    Main function to k-fold experiment on the S&P 500 reconstruction/prediction
    :return:
    """
    # Load data into DataFrame
    df = pd.read_csv(args.data_path)

    # Normalize data to be in [0,1] with mean 0.5
    df, vals_scaling_metadata = normalize_data(df)

    dfs = create_batches(df, args.seq_len)

    # Generate data loaders
    train_vals, test_vals = get_train_test_data(dfs)

    # Generate k-fold sklearn object
    kf = KFold(n_splits=args.cross_val, shuffle=True,)

    best_loss = np.Inf
    best_model = None

    for train_index, test_index in kf.split(train_vals):
        
        # Get model, optimizer and loss
        model = get_model()
        optimizer = getattr(torch.optim, args.optim)(params=model.parameters(), lr=args.lr, weight_decay=args.wd)
        criterion = rec_pred_loss if args.model_type == 'LSTMAE_PRED' else torch.nn.MSELoss(reduction='sum')

        # Get train and test ids for fold k
        train_data = [train_vals[i] for i in train_index]
        test_data = [train_vals[i] for i in test_index]

        # Getnerate relevant data loaders
        train_data = torch.utils.data.DataLoader(square_dataset(train_data, vals_scaling_metadata), batch_size=args.batch_size, shuffle=True)
        test_data = torch.utils.data.DataLoader(square_dataset(test_data, vals_scaling_metadata), batch_size=args.batch_size, shuffle=True)

        # Train and Validation loop
        pred_loss_lst, train_loss_lst, val_loss = run_training_loop(criterion, train_data, test_data, model, optimizer)

        # Save better model if found
        if val_loss < best_loss:
            print(f'Found model with better validation loss: {val_loss}')
            best_loss = val_loss
            best_model = model
        
        # Plot training losses and prediction loss graphs
        plot_train_pred_losses(train_loss_lst, pred_loss_lst, description='snp500')

    return best_model, best_loss, test_vals, vals_scaling_metadata, 


def run_training_loop(criterion, k_train_iter, k_val_iter, model, optimizer):
    """
    Training loop - including training and validation
    :param criterion: loss function
    :param k_train_iter: train data loader for current fold
    :param k_val_iter: validation data loader for current fold
    :param model: model to train
    :param optimizer: optimizer to use
    :return:
    """
    # Run train & validation evaluation for epoch
    train_loss_lst, pred_loss_lst = [], []
    for epoch in range(args.epochs):
        # Train loop
        train_loss, _, pred_loss = train_model(criterion, epoch, model, args.model_type,
                                               optimizer, k_train_iter, args.batch_size,
                                               args.grad_clipping, args.log_interval)
        train_loss_lst.append(train_loss)
        pred_loss_lst.append(pred_loss)
        val_loss, val_acc = eval_model(criterion, model, args.model_type, k_val_iter, epoch, mode='Val')
    return pred_loss_lst, train_loss_lst, val_loss


def get_model():
    """
    Generate the requested model instance
    :return: Pytorch model instance
    """
    # Generate model
    if args.model_type == 'LSTMAE':
        print('Creating LSTM AE model')
        model = LSTMAE(input_size=args.input_size, hidden_size=args.hidden_size, dropout_ratio=args.dropout,
                       seq_len=args.seq_len)
    elif args.model_type == 'LSTMAE_PRED':
        print('Creating LSTM AE with Predictor')
        model = LSTMAEPRED(input_size=args.input_size, hidden_size=args.hidden_size, dropout_ratio=args.dropout,
                           seq_len=args.seq_len)
    elif  args.model_type == 'LSTMAE_RECON':
        model = LSTMAE_RECON(args.seq_len, args.features, args.batch_size, args.embedding_size)
    else:
        raise NotImplementedError(f'Selected model type is not implemented: {args.model_type}')
    model = model.to(device)
    return model


def get_train_test_data(dfs, split_ratio=(3/4)):
    """
    Function to create the initial Train & Test split. The train will be further split in the cross validation.
    The test will be used for the final evaluation.
    In addition, the function calculate the normalization which is done on each stock individually.
    :param split_ratio: split ratio to create train/test data
    :return: Train and Test dataFrames
    """
    #random.shuffle(dfs)
    train_vals = dfs[:int(split_ratio * len(dfs))]
    test_vals = dfs[int(split_ratio * len(dfs)):]

    return train_vals, test_vals

def create_batches(data, seq_len):
    batches = []
    idx = 0
    while(idx <= len(data)-1):
        batches.append(data[idx:idx+seq_len])
        idx+= seq_len
    return batches

def normalize_data(df):
    """
    Function to normalize the data in [0,1] range with mean 0.5
    :param df: raw loaded data
    :return: data after normalization + dictionary of scaling objects for each stock
    """
    # Normalize and center each sequence around 0.5
    vals_scaling_metadata = {}
    g_mean = df[args.val_type].mean()
    vals = df[args.val_type] + (0.5 - g_mean)
    scaler = preprocessing.MinMaxScaler().fit(pd.DataFrame(vals))
    vals = scaler.transform(pd.DataFrame(vals))
    vals_scaling_metadata = {'mean': g_mean, 'scaler': scaler}
    df = vals

    return df, vals_scaling_metadata


def plot_images(model, test_iter, num_to_plot=3):
    """
    Plots the original vs. reconstructed graphs
    :param model: trained model to reconstruct stock price graphs
    :param test_iter: test data loader
    :param num_to_plot: number of random stocks to plot
    :return:
    """
    model.eval()

    test_df = test_iter.dataset.df
    test_tickers = test_iter.dataset.tickers_lst

    for i in range(num_to_plot):
        ticker = test_tickers[i]

        dates = test_df[test_df.symbol == ticker].date.tolist()
        # dates = [str(dt.datetime.strptime(d, '%Y-%m-%d').date()) for d in dates]

        data = torch.FloatTensor(test_df[test_df.symbol == ticker][args.price_type].to_list()).reshape(1, -1, 1)
        rec_data = model(data.to(device))
        if len(rec_data) > 1:
            rec_data = rec_data[0]

        # Inverse transform prices
        scaler = test_iter.dataset.tickers_scaling_metadata[ticker]['scaler']
        inversed_data = scaler.inverse_transform([data.squeeze().tolist()])
        inversed_rec_data = scaler.inverse_transform([rec_data.squeeze().tolist()])

        # Remove 0.5 centering from data
        group_mean = test_iter.dataset.tickers_scaling_metadata[ticker]['mean']
        inversed_data -= (0.5 - group_mean)
        inversed_rec_data -= (0.5 - group_mean)
        
        # orig = data[i].squeeze()
        plt.figure()
        # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=101))
        plt.xticks([i for i in range(0, len(dates), 100)], [dates[i] for i in range(0, len(dates), 100)])
        plt.plot(dates, inversed_data.squeeze(), color='g', label='Original signal')
        plt.plot(dates, inversed_rec_data.squeeze(), color='r', label='Reconstructed signal')
        plt.xlabel('Time')
        plt.ylabel('Signal Value')
        plt.legend()
        plt.title(f'Original and Rec for test example #{i + 1} {ticker}')
        plt.gcf().autofmt_xdate()
        plt.savefig(f'S&P orig vs rec{i + 1} {ticker}')
        plt.show()


def plot_train_pred_losses(train_loss_lst, prediction_loss_lst, description):
    """
    Plots the training loss graph vs. epoch and the prediction loss grpah vs. epoch.
    :param train_loss_lst: list of all train loss recorded from the training process
    :param prediction_loss_lst: list of all prediction losses recorded from the training process
    :param description: additional description to add to the plots
    :return:
    """
    epochs = [t for t in range(len(train_loss_lst))]

    plt.figure()
    plt.plot(epochs, prediction_loss_lst, color='r', label='Train prediction loss')
    plt.xticks([i for i in range(0, len(epochs) + 1, int(len(epochs) / 10))])
    plt.xlabel('Epochs')
    plt.ylabel(f'{description} Prediction loss value')
    plt.legend()
    plt.title(f'{description} Prediction loss vs. epochs')
    plt.savefig(f'{description}_prediction_graphs.png')
    plt.show()

    plt.figure()
    plt.plot(epochs, train_loss_lst, color='r', label='Train loss')
    plt.xticks([i for i in range(0, len(epochs) + 1, int(len(epochs) / 10))])
    plt.xlabel('Epochs')
    plt.ylabel(f'{description} Train loss value')
    plt.legend()
    plt.title(f'{description} Train loss vs. epochs')
    plt.savefig(f'{description}_train_loss_graphs.png')
    plt.show()


if __name__ == '__main__':
    main()
