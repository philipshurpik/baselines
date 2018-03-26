from box import Box

config = Box({
    "data": {
        "file_name": "data/SINMODEL1_256_delta0002.csv",
        "episode_duration": 480,
        "amplitude": 200
    },
    "seed": 42,
    "model": {
        "stocks_number": 1,
        "window_size": 120
    },
})
