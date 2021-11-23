import torch
import torch.utils.data
import os
import datetime
import time
import torch.utils.data as data_utils

import bs4 as bs
import numpy as np
import pandas as pd
import requests
import yfinance as yf


class MyDataset():

    def __init__(self, device='cpu', batch_size=16):
        self.url = 'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        self.stocks_fname = "sp500_closefull.csv"
        self.start = datetime.datetime(2010, 1, 1)
        self.stop = datetime.datetime.now()
        self.Ntest = 1000
        self.now = time.time()
        self.device = device
        self.batch_size = batch_size

    def get_loaders(self) -> pd.DataFrame:

        start = self.start
        end = self.stop

        if not os.path.isfile(self.stocks_fname):
            resp = requests.get(self.url)
            soup = bs.BeautifulSoup(resp.text, 'lxml')
            table = soup.find('table', {'class': 'wikitable sortable'})
            tickers = []

            for row in table.findAll('tr')[1:]:
                ticker = row.findAll('td')[0].text
                tickers.append(ticker)

            tickers = [s.replace('\n', '') for s in tickers]
            data = yf.download(tickers, start=start, end=end)
            data['Adj Close'].to_csv(self.stocks_fname)

        df0 = pd.read_csv(self.stocks_fname, index_col=0, parse_dates=True)

        df_spy = yf.download("SPY", start=start, end=end)

        df_spy = df_spy.loc[:, ['Adj Close']]

        df_spy.columns = ['SPY']

        df0 = pd.concat([df0, df_spy], axis=1)

        df0.dropna(axis=0, how='all', inplace=True)
        print("Dropping columns due to nans > 50%:",
              df0.loc[:, list((100 * (df0.isnull().sum() / len(df0.index)) > 50))].columns)
        df0 = df0.drop(df0.loc[:, list((100 * (df0.isnull().sum() / len(df0.index)) > 50))].columns, 1)
        df0 = df0.ffill().bfill()

        print("Any columns still contain nans:", df0.isnull().values.any())

        df_returns = pd.DataFrame()
        for name in df0.columns:
            df_returns[name] = np.log(df0[name]).diff()

        df_returns.dropna(axis=0, how='any', inplace=True)
        # Convert returns into 0/1
        df_returns.SPY = [1 if spy > 0 else 0 for spy in df_returns.SPY]

        #  =====================================
        train_data = df_returns.iloc[:-self.Ntest]
        validate_data = df_returns.iloc[len(df_returns) - self.Ntest:(len(df_returns)) - int(self.Ntest / 2)]
        test_data = df_returns.iloc[-int(self.Ntest / 2):]

        # All data MUST BE FLOAT It´s due to PyTorch ! (no ints here, neither labels are ints)
        train_labels = torch.tensor(train_data.SPY.values).float().unsqueeze(1)
        train_features = torch.tensor(train_data.drop('SPY', axis=1).values).float()
        #
        validate_labels = torch.tensor(validate_data.SPY.values).float().unsqueeze(1)
        validate_features = torch.tensor(validate_data.drop('SPY', axis=1).values).float()
        #
        test_labels = torch.tensor(test_data.SPY.values).float().unsqueeze(1)
        test_features = torch.tensor(test_data.drop('SPY', axis=1).values).float()

        # ==============================
        training_data = data_utils.TensorDataset(train_features, train_labels)
        # training_data=data_utils.TensorDataset(train_features)
        test_data = data_utils.TensorDataset(test_features, test_labels)
        validate_data = data_utils.TensorDataset(validate_features, validate_labels)

        self.column_count = len(training_data[0][0])

        loaders = (data_utils.DataLoader(training_data, batch_size=self.batch_size),
                   data_utils.DataLoader(test_data, batch_size=self.batch_size),
                   data_utils.DataLoader(validate_data, batch_size=self.batch_size))

        return loaders