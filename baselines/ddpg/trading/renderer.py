import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates, ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


def render_sticks(ax, quotes, width=1000, colorup='#00FF00', colordown='#FF0000', alpha=0.8):
    for q in quotes:
        t, open, high, low, close, volume, buy_trade, sell_trade = q
        timestamp = dates.date2num(t)

        if close >= open:
            color = colorup
            lower = open
            height = close - open
        else:
            color = colordown
            lower = close
            height = open - close

        vline = Line2D(
            xdata=(timestamp, timestamp), ydata=(low, high),
            color=color,
            linewidth=0.5,
            antialiased=True,
        )
        rect = Rectangle(
            xy=(timestamp, lower),
            width=0.5/len(quotes),
            height=height,
            facecolor=color,
            edgecolor=color,
        )
        rect.set_alpha(alpha)
        rect.set_linewidth((width * 0.8)/len(quotes))
        ax.add_line(vline)
        ax.add_patch(rect)


def render_trades(ax, timestamps, prices, marker, color):
    data = np.array(list(zip(timestamps, prices)))
    x = data[data[:, 1] != 0][:, 0]
    y = data[data[:, 1] != 0][:, 1]
    ax.scatter(x, y, marker=marker, color=color)


def render(values, title, scale=1):
    fig, ax = plt.subplots(figsize=(20*scale, 8*scale))
    render_sticks(ax, values, width=20*50*scale)
    timestamps = dates.date2num(values[:, 0])
    render_trades(plt, timestamps, prices=values[:, 6], marker='^', color='g')
    render_trades(plt, timestamps, prices=values[:, 7], marker='v', color='r')
    ax.autoscale_view()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(20))
    ax.xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=40)
    plt.title(title)
    plt.grid()
    plt.show()
