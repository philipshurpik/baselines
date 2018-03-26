import os
from trading.TradingEnv import TradingEnv
from trading.sin_config import config
env = TradingEnv(csv_name=config.data.file_name, window_size=config.model.window_size,
                 episode_duration=config.data.episode_duration, amplitude=config.data.amplitude)
env.render()
