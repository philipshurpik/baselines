from box import Box

config = Box({
    "data": {
        "file_name": "data/BTCETH60.csv",
        "episode_duration": 480
    },
    "seed": 42,
    "model": {
        "stocks_number": 1,
        "window_size": 30
    }
})
