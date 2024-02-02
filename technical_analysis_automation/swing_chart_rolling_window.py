import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_rolling_window(data_set: np.array, current_index: int, time_radius: int, is_top: bool) -> bool:
    if current_index < time_radius * 2 + 1:
        return False

    result_flag = True
    comparison_operator = np.greater if is_top else np.less

    time_span_starting_index = current_index - time_radius
    time_span_starting_value = data_set[time_span_starting_index]

    for loop_index in range(1, time_radius + 1):
        if comparison_operator(data_set[time_span_starting_index + loop_index], time_span_starting_value) or \
                comparison_operator(data_set[time_span_starting_index - loop_index], time_span_starting_value):
            result_flag = False
            break

    return result_flag


def rw_extremes(data_set: np.array, time_radius: int):
    # Rolling window local tops and bottoms
    tops = []
    bottoms = []
    for loop_index in range(len(data_set)):
        if calculate_rolling_window(data_set, loop_index, time_radius, True):
            # top[0] = confirmation index
            # top[1] = index of top
            # top[2] = price of top
            top = [loop_index, loop_index - time_radius, data_set[loop_index - time_radius]]
            tops.append(top)

        if calculate_rolling_window(data_set, loop_index, time_radius, False):
            # bottom[0] = confirmation index
            # bottom[1] = index of bottom
            # bottom[2] = price of bottom
            bottom = [loop_index, loop_index - time_radius, data_set[loop_index - time_radius]]
            bottoms.append(bottom)

    return tops, bottoms


def main():
    data = pd.read_csv('.././data/BTCUSDT86400.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')

    tops, bottoms = rw_extremes(data['close'].to_numpy(), 10)
    data['close'].plot()
    idx = data.index
    for top in tops:
        plt.plot(idx[top[1]], top[2], marker='o', color='green')

    for bottom in bottoms:
        plt.plot(idx[bottom[1]], bottom[2], marker='o', color='red')

    plt.show()


if __name__ == "__main__":
    main()

# Scipy implementation (faster but use with care to not cheat with future data)
# import scipy
# arr = data['close'].to_numpy()
# bottoms = scipy.signal.argrelextrema(arr, np.less, order=3)
# tops = scipy.signal.argrelextrema(arr, np.greater, order=3)
