import numpy as np
import pandas as pd


class TradingSimulator(object):
    def __init__(self, csv_name, model_config, data_config):
        df = pd.read_csv(csv_name, parse_dates=[data_config.date_columns])
        df = df[~np.isnan(df['Close'])].set_index(pd.DatetimeIndex(df[data_config.index_column]))
        if data_config.start_date is not None:
            df = df.iloc[data_config.start_date:]

        value_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[value_columns] = df[value_columns].apply(pd.to_numeric)

        # temp stuff for sin simulation
        if data_config.amplitude is not None:
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = self._sin_add_amplitude(df[col], data_config.amplitude)
        self.data = df
        self.data_values = df[value_columns]
        self.stock_name = df['Currency'][0]
        self.date_time = self.data.index
        self.count = self.data.shape[0]
        self.window_size = model_config.window_size
        self.episode_duration = data_config.episode_duration
        self.train_end_index = int(data_config.train_split * self.count)
        self.start_index, self.end_index = self._get_start_end_index()
        self.current_index = self.start_index
        self.step_number = 0
        if model_config.type == 'conv':
            self.states = np.array([
                self._normalize_column(df['Open'], normalize=True).values,
                self._normalize_column(df['High'], normalize=True).values,
                self._normalize_column(df['Low'], normalize=True).values,
                self._normalize_column(df['Close'], normalize=True).values,
                self._normalize_column(df['Volume'].fillna(0), normalize=True).values,
            ])
        else:
            self.states = np.array([self._normalize_column(df['Close'], normalize=True).values])

        self.features_number = self.states.shape[0]
        self.states = self.states.T
        self.reset()

    @staticmethod
    def seed(seed):
        np.random.seed(seed)

    @staticmethod
    def _sin_add_amplitude(df_column, amplitude):
        return df_column + ((np.mean(df_column) - df_column) * amplitude)

    def _normalize_column(self, df_column, normalize):
        column = df_column.copy().pct_change().replace(np.inf, np.nan).fillna(0)
        eps = np.finfo(np.float32).eps
        column_n = (column - np.array(column.mean())) / np.array(column.std() + eps)
        return column if not normalize else column_n

    def _get_current_window(self):
        reversed_window = self.states[self.current_index - self.window_size: self.current_index][::-1]
        return reversed_window.reshape(1, self.window_size, self.features_number)

    def _get_start_end_index(self, train_mode=True):
        train_index = np.random.randint(self.window_size, self.train_end_index - self.episode_duration - 1)
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
        done = True if self.current_index >= self.end_index - 1 else False
        if not done:
            self.step_number += 1
            self.current_index += 1
        return self._get_current_window(), done

    def get_episode_values(self):
        time = self.date_time[self.start_index: self.end_index].to_pydatetime().reshape(-1, 1)
        values = self.data_values.values[self.start_index:self.end_index, :]
        return np.concatenate((time, values), axis=1)
