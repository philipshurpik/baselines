from box import Box

config = Box({
    "data": {
        "file_name": "data/USDBTC60.csv",
        "episode_duration": 480,
        "date_columns": ["Date", "Time"],
        "index_column": "Date_Time",
        "start_date": None,
        "train_split": 0.8,
        "initial_cash": 1000,
        "commission": 0.0025,
        "amplitude": None,
    },
    "seed": 42,
    "model": {
        # type - conv | fc | lstm
        "type": "conv",
        "stocks_number": 1,
        "window_size": 120,
        "save_folder": None
    },
})
