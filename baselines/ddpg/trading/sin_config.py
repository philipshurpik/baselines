from box import Box

config = Box({
    "data": {
        "file_name": "data/SINMODEL1_256_delta0002.csv",
        "episode_duration": 480,
        "amplitude": 10
    },
    "seed": 42,
    "model": {
        # type - conv | fc | lstm
        "type": "fc",
        "stocks_number": 1,
        "window_size": 120
    },
})
