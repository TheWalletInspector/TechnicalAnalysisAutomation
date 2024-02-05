import numpy as np
import pandas as pd
import mplfinance as mpf
from matplotlib import pyplot as plt


def is_local_swing_extreme(data_set: np.array, current_index: int, time_radius: int, is_top: bool) -> bool:
    # If the current index does not have enough data points to form a rolling window, return False
    if current_index < time_radius * 2 + 1:
        return False

    comparison_operator = np.greater if is_top else np.less

    time_span_starting_index = current_index - time_radius
    time_span_starting_value = data_set[time_span_starting_index]

    # If any of the data points within the time span are greater than the starting value, return False (not a local extreme)
    return all(
        not comparison_operator(data_set[time_span_starting_index + loop_index], time_span_starting_value) and
        not comparison_operator(data_set[time_span_starting_index - loop_index], time_span_starting_value)
        for loop_index in range(1, time_radius + 1)
    )


def detect_swing_extremes(data_set: np.array, time_radius: int):
    local_tops = [
        [loop_index, loop_index - time_radius, data_set[loop_index - time_radius]]
        for loop_index in range(len(data_set))
        if is_local_swing_extreme(data_set, loop_index, time_radius, True)
    ]

    local_bottoms = [
        [loop_index, loop_index - time_radius, data_set[loop_index - time_radius]]
        for loop_index in range(len(data_set))
        if is_local_swing_extreme(data_set, loop_index, time_radius, False)
    ]

    return (
        pd.DataFrame(local_tops, columns=["confirmation_index", "index_of_swing", "price_of_swing"]),
        pd.DataFrame(local_bottoms, columns=["confirmation_index", "index_of_swing", "price_of_swing"])
    )


def main():
    data = pd.read_csv('.././data/BTCUSDT86400.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')

    swing_tops, swing_bottoms = detect_swing_extremes(data['close'].to_numpy(), 10)

    # Plotting as a trading bar chart
    fig, axlist = mpf.plot(data,
                           type='candle',  # candlestick chart
                           style='yahoo',  # chart style
                           addplot=[
                               mpf.make_addplot(swing_tops['price_of_swing'], color='green', marker='^', markersize=8),
                               mpf.make_addplot(swing_bottoms['price_of_swing'], color='red', marker='v', markersize=8)
                           ],
                           volume=False,
                           figratio=(10, 6),  # figure size
                           title='BTCUSDT Trading Chart',
                           ylabel='Price',
                           ylabel_lower='Volume'
                           )

    # Display legend
    axlist[0].legend()

    # Show the plot
    plt.show()


if __name__ == '__main__':
    main()
