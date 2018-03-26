import os
from trading.TradingEnv import TradingEnv
from trading.sin_config import config
env_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), config.data.file_name)
env = TradingEnv(csv_name=env_filename, model_config=config.model, data_config=config.data)
env.render()
