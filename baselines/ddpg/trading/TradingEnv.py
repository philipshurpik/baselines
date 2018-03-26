import math
import os
import sys
import collections

import gym
import numpy as np
from gym import spaces

from .TradingSimulator import TradingSimulator
from .TradingStats import TradingStats
from .renderer import render

"""
we can have 3 environment versions:
1) the simplest one:
Discrete Action space of 2 elements - Long and Short
Only 1 bitcoin at start and we can buy or sell all amount of ETC that can be done on bitcoin
Calculate reward on each step
Use comission for each trade
Close in the end or if dropdown is bigger than 10% with huge punishment   

2) discrete with more choices: 
Discrete Action space of 4 elements - Long, Hold, Short and Close
Amount of action can be discrete as half of full stake (like 0.5 btc from 1 btc)

3) Continuous action space
Continuous space from -1 to 1 that define exact amount to buy or sell 
First value: Use tanh to define action value from -1 (short) to 1 (long)

We can somehow control do we need to execute action or not because small action will make us pay comissions on each trade
Second value: Use tanh to define how sure are we to make trade: -1.0..0 - no trade, 0..1 - make trade

The 3 Version was implemented here:  
"""
DECIMAL_SIGNS = 6
rnd = lambda x: round(x, DECIMAL_SIGNS)
JournalAction = collections.namedtuple('Action', ('index', 'date', 'size', 'price', 'meta'))
SIZE_FIELD_INDEX = JournalAction._fields.index('size')


class TradingEnv(gym.Env):
    COMMISSION = 0.0025

    def __init__(self, csv_name, window_size, episode_duration, save_folder=None, initial_cash=100, amplitude=None,
                 date_columns=["Date", "Time"], index_column="Date_Time", start_date=None, verbose=0, train_mode=True):
        self.simulator = TradingSimulator(csv_name=csv_name, date_columns=date_columns, index_column=index_column, amplitude=amplitude,
                                          start_date=start_date, window_size=window_size, episode_duration=episode_duration)
        self.verbose = verbose
        self.train_mode = train_mode
        self.window_size = window_size
        self.features_number = self.simulator.features_number
        self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        high = np.ones((1, self.window_size, self.features_number)) if self.features_number > 1 else np.ones(window_size)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.min_position_size = 0.01
        self.initial_cash = initial_cash
        self.remaining_cash_value = initial_cash
        self.max_cash_value = initial_cash
        self.max_draw_down = 0
        self.position_value = 0
        self.position_size = 0
        self.total_reward = 0
        self.cumulative_step_reward = 0
        self.journal = []
        self.train_stats = TradingStats(os.path.join(save_folder, "ddpg_train_stats.csv") if save_folder else None)
        self.test_stats = TradingStats(os.path.join(save_folder, "ddpg_test_stats.csv") if save_folder else None)
        self.reset()

    def seed(self, seed=None):
        np.random.seed(seed)
        self.simulator.seed(seed)

    def step(self, action):
        close_price = self.simulator.data.iloc[self.simulator.current_index]["Close"]
        date = self.simulator.date_time[self.simulator.current_index]
        action_meta = self._calculate_step(action=action, close_price=close_price, position_size=self.position_size,
                                           position_value=self.position_value, remaining_cash=self.remaining_cash_value)
        if abs(action_meta["size_diff"]) >= self.min_position_size:
            self.journal.append(JournalAction(index=self.simulator.step_number, date=date, size=action_meta['size_diff'], price=close_price, meta=action_meta))
            if self.verbose:
                self.print_action(action_meta, close_price, date)

        sim_state, done = self.simulator.step()
        self.remaining_cash_value = action_meta["remaining_cash"]
        self.position_value = action_meta["value"]
        self.position_size = action_meta["size"]
        self.max_cash_value = max(action_meta["total_cash"], self.max_cash_value)
        self.max_draw_down = max((1 - action_meta["total_cash"] / self.max_cash_value), self.max_draw_down)
        self.total_reward = self.total_reward + action_meta["reward"]

        step_reward = action_meta["reward"]

        if self.max_draw_down > 0.50:
            done = True
            step_reward -= 0.1 * self.initial_cash * self.simulator.episode_duration / self.simulator.step_number
            print("Exceed maximum drawdown: Max cash %.2f | Current cash: %.2f" % (self.max_cash_value, action_meta["total_cash"]))

        # env_state = self._get_env_state(reward=step_reward, action=action)
        all_state = self._reshape(sim_state)
        return all_state, step_reward, done, {"date": self.simulator.date_time[self.simulator.current_index]}

    def reset(self):
        if self.simulator.step_number > 0:
            self.print_summary()
        self.remaining_cash_value = self.initial_cash
        self.max_cash_value = self.initial_cash
        self.max_draw_down = 0
        self.position_value = 0
        self.position_size = 0
        self.total_reward = 0
        self.cumulative_step_reward = 0
        self.journal = []

        # env_state = self._get_env_state(0, [0, 0])
        sim_state, done = self.simulator.reset(train_mode=self.train_mode)
        return self._reshape(sim_state)   # [env_state, sim_state]

    def _reshape(self, state):
        return state.reshape(1, self.window_size, self.features_number) if self.features_number > 1 \
            else state.reshape(self.window_size)

    def _get_env_state(self, reward, action):
        total_cash = self.remaining_cash_value - self.position_value
        position = 1 if self.position_value > 0 else 0
        return np.array([total_cash, self.remaining_cash_value, position]).reshape(-1, 3)

    def _calculate_step(self, action, close_price, position_value, position_size, remaining_cash):
        # Use tanh to define action value from -1 (short) to 1 (long)
        action_amount = min(1, max(-1, action[0]))
        # Use sigmoid to define how sure are we to make trade: -1..0 - no trade, 0..1 - make trade
        hold = action[1] < 0
        hold_data = self._calculate_step_hold(close_price, position_value, position_size, remaining_cash)
        if hold:
            return hold_data
        else:
            self.cumulative_step_reward = 0
            return self._calculate_step_position(action_amount, close_price, hold_data)

    @staticmethod
    def _calculate_step_hold(close_price, position_value, position_size, remaining_cash):
        if position_size > 0:
            new_position_value = position_size * close_price
            step_reward = new_position_value - position_value
        elif position_size < 0:
            new_position_value = position_size * close_price  # -7 = -2 * 3.5
            step_reward = new_position_value - position_value  # -1 = -7 - -6
        else:
            new_position_value = position_value
            step_reward = 0
        return {
            "value": rnd(new_position_value),
            "value_diff": rnd(new_position_value - position_value),
            "size": rnd(position_size),
            "size_diff": 0,
            "remaining_cash": rnd(remaining_cash),
            "reward": rnd(step_reward),
            "total_cash": rnd(remaining_cash + new_position_value)
        }

    @staticmethod
    def _calculate_step_position(action_amount, close_price, hold_data):
        hold_value, hold_size, hold_remaining, hold_total, hold_reward = \
            hold_data["value"], hold_data["size"], hold_data["remaining_cash"], hold_data["total_cash"], hold_data["reward"]

        value_wo_comission = hold_total * action_amount
        value_wo_comission_diff = value_wo_comission - hold_value
        value_diff = value_wo_comission_diff / (1 + TradingEnv.COMMISSION)
        size_diff = value_diff / close_price

        value = hold_value + value_diff
        size = hold_size + size_diff

        commission = abs(value_diff - value_wo_comission_diff)
        remaining_cash = hold_remaining - value_diff - commission
        total_cash = hold_total - commission
        total_step_reward = hold_reward - commission

        if math.isnan(rnd(value)) or math.isnan(rnd(value_diff)) or math.isnan(rnd(size)) or math.isnan(rnd(size_diff)) or math.isnan(rnd(remaining_cash)) or math.isnan(rnd(total_cash)) or math.isnan(rnd(total_step_reward)):
            print("\n\n\n\n\nlalala is nan\n\n\n\n\n")

        return {
            "value": rnd(value),
            "value_diff": rnd(value_diff),
            "size": rnd(size),
            "size_diff": rnd(size_diff),
            "remaining_cash": rnd(remaining_cash),
            "total_cash": rnd(total_cash),
            "reward": rnd(total_step_reward)
        }

    def print_action(self, action_meta, price, date):
        order_type = "Buy " if action_meta["size_diff"] > 0 else "Sell"
        print("%s | %s | Reward: %.6f | Size: %.4f %s | Price: %.4f | Position value: %.4f | Position size: %.4f | Total cash: %.6f" %
              (date, order_type, action_meta["reward"], action_meta["size_diff"], self.simulator.stock_name, price, action_meta["value"],
               action_meta["size"], action_meta["total_cash"]))

    def reset_train_mode(self, train_mode):
        self.train_mode = train_mode
        return self.reset()

    def print_summary(self):
        sim = self.simulator
        start_date = sim.date_time[sim.start_index]
        end_date = sim.date_time[sim.end_index]
        start_p = sim.data.loc[start_date]["Close"]
        end_p = sim.data.loc[end_date]["Close"]
        buy_hold_value = self.initial_cash * (end_p/start_p) - self.initial_cash
        train_or_test = "Train" if self.train_mode else "Test"
        total_cash = self.remaining_cash_value + self.position_value

        action_sizes = np.array(list(zip(*self.journal))[SIZE_FIELD_INDEX]) if len(self.journal) > 0 else np.array([])
        long_actions = action_sizes[action_sizes > 0]
        short_actions = action_sizes[action_sizes < 0]
        print("%s\t\t| Start: %s\t| End: %s\t| Buy&Hold: %.4f\t| Longs: %d\t| Shorts: %d\t| Ep. Reward %.4f\t" %
              (sim.stock_name, start_date, end_date, buy_hold_value, len(long_actions), len(short_actions), self.total_reward))
        print("%s\t| Operations: %d\t| Max Drawdown: %.4f\t| Max cash: %.4f\t| Final cash: %.4f" %
              (train_or_test, len(self.journal), self.max_draw_down, self.max_cash_value, total_cash))
        stats = self.train_stats if self.train_mode else self.test_stats
        stats.add(self.total_reward, buy_hold_value, self.max_draw_down, 1, len(self.journal))
        stats.print_latest()
        self.render()

    def render(self, mode='human'):
        title = self.simulator.stock_name + (" Train" if self.train_mode else " Test")
        values = self.simulator.get_episode_values()
        trades = np.zeros((len(values), 2))
        for trade in self.journal:
            trade_type = 0 if trade.size > 0 else 1
            trades[trade.index][trade_type] = trade.price
        trade_values = np.concatenate((values, trades), axis=1)
        # Scale for PyCharm debugger SciView
        scale = 0.5 if sys.gettrace() else 1
        render(trade_values, title=title, scale=scale)
