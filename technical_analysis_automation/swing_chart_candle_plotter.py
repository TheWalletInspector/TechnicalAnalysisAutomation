import logging

import mplfinance as mpf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def is_data_point_a_local_swing(data_set: np.array, current_index: int, time_radius: int, is_top: bool) -> bool:
    """
    Check if the current index is a local swing based on the data set and time radius.
    """
    current_index = current_index - 1
    if current_index < time_radius:
        print("")
        print(f"current_index: {current_index} is less than time_radius * 2 + 1: {time_radius * 2 + 1}")
        logging.info(f"current_index: {current_index} does not have enough data points to form a rolling window")
        return False

    comparison_operator = np.greater if is_top else np.less

    time_span_starting_index = current_index - time_radius
    time_span_starting_value = data_set[time_span_starting_index]

    print(f"current_index: {current_index}")
    print(f"time_span_starting_index: {time_span_starting_index}")
    print(f"time_span_starting_value: {time_span_starting_value}")

    # If any of the data points within the time span are greater than the starting value, return False (not a local extreme)
    for loop_index in range(1, time_radius + 1):
        condition1 = not comparison_operator(data_set[time_span_starting_index + loop_index], time_span_starting_value)
        print(f"data_set[time_span_starting_index + loop_index]: {data_set[time_span_starting_index + loop_index], time_span_starting_value}")

        condition2 = not comparison_operator(data_set[time_span_starting_index - loop_index], time_span_starting_value)
        print(f"data_set[time_span_starting_index + loop_index]: {data_set[time_span_starting_index + loop_index], time_span_starting_value}")

        print(f"loop_index: {loop_index}, condition1: {condition1}, condition2: {condition2}")


    return all(
        not comparison_operator(data_set[time_span_starting_index + loop_index], time_span_starting_value) and
        not comparison_operator(data_set[time_span_starting_index - loop_index], time_span_starting_value)
        for loop_index in range(1, time_radius + 1)
    )


def detect_swing_extremes_across_data_set(data_set: np.array, time_radius: int) -> (pd.DataFrame, pd.DataFrame):
    local_tops = [
        [index_of_bar_to_test, index_of_bar_to_test - time_radius, data_set[index_of_bar_to_test - time_radius]]
        for index_of_bar_to_test in range(len(data_set))
        if is_data_point_a_local_swing(data_set=data_set, current_index=index_of_bar_to_test, time_radius=time_radius,
                                       is_top=True)
    ]

    local_bottoms = [
        [loop_index, loop_index - time_radius, data_set[loop_index - time_radius]]
        for loop_index in range(len(data_set))
        if is_data_point_a_local_swing(data_set, loop_index, time_radius, False)
    ]

    return (
        pd.DataFrame(local_tops, columns=["confirmation_index", "index_of_swing", "price_of_swing"]),
        pd.DataFrame(local_bottoms, columns=["confirmation_index", "index_of_swing", "price_of_swing"])
    )


def main() -> None:
    data = pd.read_csv('.././data/BTCUSDT86400.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')

    swing_tops, swing_bottoms = detect_swing_extremes_across_data_set(data['close'].to_numpy(), 10)

    # Create new series for the swing highs and lows
    data['swing_high'] = np.nan
    data['swing_low'] = np.nan
    data.loc[swing_tops['index_of_swing'], 'swing_high'] = swing_tops['price_of_swing']
    data.loc[swing_bottoms['index_of_swing'], 'swing_low'] = swing_bottoms['price_of_swing']

    # Plotting as a trading bar chart
    fig, axlist = mpf.plot(data,
                           type='bar',  # candlestick chart
                           style='yahoo',  # chart style
                           addplot=[
                               mpf.make_addplot(data['swing_high'], color='green', linestyle='solid'),
                               mpf.make_addplot(data['swing_low'], color='red', linestyle='solid'),
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
