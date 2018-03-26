from box import Box

config = Box({
    "data": {
        "file_name": "data/SINMODEL1_256_delta0002.csv",
        "episode_duration": 480
    },
    "seed": 1,
    "model": {
        "stocks_number": 1,
        "window_size": 120
    },
})
