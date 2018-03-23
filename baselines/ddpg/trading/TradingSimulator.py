import numpy as np
import pandas as pd


class TradingSimulator(object):
    def __init__(self, csv_name, start_date, date_columns, index_column, window_size,
                 episode_duration, train_split=0.8, normalize=True):
        df = pd.read_csv(csv_name, parse_dates=[date_columns])
        df = df[~np.isnan(df['Close'])].set_index(pd.DatetimeIndex(df[index_column]))
        if start_date is not None:
            df = df.iloc[start_date:]

        value_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[value_columns] = df[value_columns].apply(pd.to_numeric)
        self.data = df
        self.data_values = df[value_columns]
        self.stock_name = df.iloc[0, 0]
        self.date_time = self.data.index
        self.count = self.data.shape[0]
        self.window_size = window_size
        self.episode_duration = episode_duration
        self.train_end_index = int(train_split * self.count)
        self.start_index, self.end_index = self._get_start_end_index()
        self.current_index = self.start_index
        self.step_number = 0
        self.features_number = 2

        self.states = np.array([
            #self._normalize_column(df['Open'], normalize).values,
            #self._normalize_column(df['High'], normalize).values,
            #self._normalize_column(df['Low'], normalize).values,
            self._normalize_column(df['Close'], normalize).values,
            self._normalize_column(df['Volume'].fillna(0), normalize).values,
        ]).reshape((-1, self.features_number))
        self.reset()

    @staticmethod
    def seed(seed):
        np.random.seed(seed)

    def _normalize_column(self, df_column, normalize):
        column = df_column.copy().pct_change().replace(np.inf, np.nan).fillna(0)
        eps = np.finfo(np.float32).eps
        column_n = (column - np.array(column.mean())) / np.array(column.std() + eps)
        return column if not normalize else column_n

    def _get_current_window(self):
        reversed_window = self.states[self.current_index - self.window_size: self.current_index][::-1]
        return reversed_window.reshape(1, self.window_size, self.features_number)

    def _get_start_end_index(self, train_mode=True):
        train_index = self.window_size  # np.random.randint(self.window_size, self.train_end_index - self.episode_duration - 1)
        test_index = np.random.randint(self.train_end_index + self.window_size, self.count - self.episode_duration - 1)
        start_index = train_index if train_mode else test_index
        end_index = start_index + self.episode_duration
        return start_index, end_index

    def reset(self, train_mode=True):
        self.start_index, self.end_index = self._get_start_end_index(train_mode)
        self.current_index = self.start_index
        self.step_number = 0
        return self._get_current_window(), False

    def step(self):
        done = True if self.current_index >= self.end_index else False
        if not done:
            self.step_number += 1
            self.current_index += 1
        return self._get_current_window(), done

    def get_episode_values(self):
        time = self.date_time[self.start_index: self.end_index].to_pydatetime().reshape(-1,1)
        values = self.data_values.values[self.start_index:self.end_index, :]
        return np.concatenate((time, values), axis=1)
