from box import Box

config = Box({
    "data": {
        "file_name": "data/SINMODEL1_256_delta0002.csv",
        "episode_duration": 480
    },
    "seed": 42,
    "model": {
        "stocks_number": 1,
        "window_size": 120
    },
    "ddpg": {
        "buffer_size": 100000,
        "batch_size": 32,
        "gamma": 0.99,
        "tau": 0.001,
        "learning_rate_actor": 0.0001,
        "learning_rate_critic": 0.001,
        "explore": 1000000.
    }
})
