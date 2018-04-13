import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates, ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


def render_sticks(ax, quotes, has_date, base_rect_coef=0.5, width=1000, colorup='#00FF00', colordown='#FF0000', alpha=0.8):
    index = 0
    rect_width = (base_rect_coef * 0.5 if len(quotes) < 100 else base_rect_coef) / len(quotes)
    for q in quotes:
        if has_date:
            open, high, low, close = q[1], q[2], q[3], q[4]
            timestamp = dates.date2num(q[0])
        else:
            open, high, low, close = q[0], q[1], q[2], q[3]
            timestamp = index

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
            width=rect_width,
            height=height,
            facecolor=color,
            edgecolor=color,
        )
        rect.set_alpha(alpha)
        rect.set_linewidth((width * 0.8)/len(quotes))
        ax.add_line(vline)
        ax.add_patch(rect)
        index += 1


def render_trades(ax, timestamps, prices, marker, color):
    data = np.array(list(zip(timestamps, prices)))
    x = data[data[:, 1] != 0][:, 0]
    y = data[data[:, 1] != 0][:, 1]
    ax.scatter(x, y, marker=marker, color=color)


def render(values, title, scale=1, base_rect_coef=0.5):
    has_trades = values.shape[1] == 8
    has_date = has_trades or values.shape[1] == 6
    fig, ax = plt.subplots(figsize=(20*scale, 8*scale))
    render_sticks(ax, values, has_date=has_date, base_rect_coef=base_rect_coef, width=20*50*scale)
    if has_trades:
        timestamps = dates.date2num(values[:, 0])
        render_trades(plt, timestamps, prices=values[:, 6], marker='^', color='g')
        render_trades(plt, timestamps, prices=values[:, 7], marker='v', color='r')
    if has_date:
        ax.xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.autoscale_view()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(20))
    plt.xticks(rotation=40)
    plt.title(title)
    plt.grid()
    plt.show()
