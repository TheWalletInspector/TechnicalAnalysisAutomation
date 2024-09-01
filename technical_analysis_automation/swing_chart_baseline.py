import logging

import mplfinance as mpf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def _is_local_extreme(data_set: np.array, current_index: int, time_radius: int, is_top: bool) -> bool:
    if current_index < time_radius:
        print("")
        print(f"current_index: {current_index} is less than time_radius: {time_radius}")
        logging.info(f"current_index: {current_index} does not have enough data points to form a rolling window")
        return False

    start = max(0, current_index - time_radius)
    end = min(len(data_set), current_index + time_radius + 1)
    window = data_set[start:end]

    comparison_operator = np.max if is_top else np.min
    is_extreme = data_set[current_index] == comparison_operator(window) and np.sum(
        window == data_set[current_index]) == 1

    return bool(is_extreme)


def detect_swing_extremes_across_data_set(data_set: np.array, time_radius: int) -> (pd.DataFrame, pd.DataFrame):
    """
    Detects swing highs and lows across the data set
    """
    local_tops = [
        [index_of_bar_to_test, data_set[index_of_bar_to_test]]
        for index_of_bar_to_test in range(len(data_set))
        if _is_local_extreme(data_set=data_set,
                             current_index=index_of_bar_to_test,
                             time_radius=time_radius,
                             is_top=True)
    ]

    local_bottoms = [
        [index_of_bar_to_test, data_set[index_of_bar_to_test]]
        for index_of_bar_to_test in range(len(data_set))
        if _is_local_extreme(data_set=data_set,
                             current_index=index_of_bar_to_test,
                             time_radius=time_radius,
                             is_top=False)
    ]

    return (
        pd.DataFrame(local_tops, columns=["index_of_swing", "price_of_swing"]),
        pd.DataFrame(local_bottoms, columns=["index_of_swing", "price_of_swing"])
    )


def main() -> None:
    data = pd.read_csv('.././data/BTCUSDT86400.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')

    swing_tops, swing_bottoms = detect_swing_extremes_across_data_set(data['close'].to_numpy(), 2)

    print("Swing Tops")
    print(swing_tops)

    print("Swing Bottoms")
    print(swing_bottoms)
    # # Create new series for the swing highs and lows
    # data['swing_high'] = np.nan
    # data['swing_low'] = np.nan
    # data.loc[swing_tops['index_of_swing'], 'swing_high'] = swing_tops['price_of_swing']
    # data.loc[swing_bottoms['index_of_swing'], 'swing_low'] = swing_bottoms['price_of_swing']
    #
    # # Plotting as a trading bar chart
    # fig, axlist = mpf.plot(data,
    #                        type='bar',  # candlestick chart
    #                        style='yahoo',  # chart style
    #                        addplot=[
    #                            mpf.make_addplot(data['swing_high'], color='green', linestyle='solid'),
    #                            mpf.make_addplot(data['swing_low'], color='red', linestyle='solid'),
    #                            mpf.make_addplot(swing_tops['price_of_swing'], color='green', marker='^', markersize=8),
    #                            mpf.make_addplot(swing_bottoms['price_of_swing'], color='red', marker='v', markersize=8)
    #                        ],
    #                        volume=False,
    #                        figratio=(10, 6),  # figure size
    #                        title='BTCUSDT Trading Chart',
    #                        ylabel='Price',
    #                        ylabel_lower='Volume'
    #                        )
    #
    # # Display legend
    # axlist[0].legend()
    #
    # # Show the plot
    # plt.show()


if __name__ == '__main__':
    main()