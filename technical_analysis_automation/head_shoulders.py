from collections import deque
from dataclasses import dataclass

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

from rolling_window import rw_top, rw_bottom


@dataclass(slots=True)
class HSPattern:
    # True if inverted, False if not. Inverted is "bullish" according to technical analysis dogma.
    inverted: bool

    # Indices of the parts of the H&S pattern
    l_shoulder: int = -1
    r_shoulder: int = -1
    l_armpit: int = -1
    r_armpit: int = -1
    head: int = -1

    # Price of the parts of the H&S pattern. _p stands for price.
    l_shoulder_price: float = -1.0
    r_shoulder_price: float = -1.0
    l_armpit_price: float = -1.0
    r_armpit_price: float = -1.0
    head_price: float = -1.0

    start_i: int = -1
    break_i: int = -1
    break_price: float = -1.0

    neck_start: float = -1.0
    neck_end: float = -1.0

    # Attributes
    neck_slope: float = -1.0
    head_width: float = -1.0
    head_height: float = -1.0
    pattern_r2: float = -1.0

    def compute_r2(self, price_data: np.array):
        """
        Computes the coefficient of determination (R^2 value) for the Head and Shoulders (H&S) pattern.

        :param price_data: The array of price data points.
        :param self: Instance of the HSPattern class representing a detected H&S pattern.
        :return: The R^2 value for the H&S pattern, indicating the goodness of fit of the pattern to the data.
        """

        def compute_line_segment(start_price: float, end_price: float, start_index: int, end_index: int) -> np.array:
            """:return: An array representing the price values along the segment."""
            segment_slope = (end_price - start_price) / (end_index - start_index)
            return start_price + np.arange(end_index - start_index) * segment_slope

        pattern_segments = [
            compute_line_segment(self.neck_start, self.l_shoulder_price, self.start_i, self.l_shoulder),
            compute_line_segment(self.l_shoulder_price, self.l_armpit_price, self.l_shoulder, self.l_armpit),
            compute_line_segment(self.l_armpit_price, self.head_price, self.l_armpit, self.head),
            compute_line_segment(self.head_price, self.r_armpit_price, self.head, self.r_armpit),
            compute_line_segment(self.r_armpit_price, self.r_shoulder_price, self.r_armpit, self.r_shoulder),
            compute_line_segment(self.r_shoulder_price, self.break_price, self.r_shoulder, self.break_i)
        ]

        pattern_prices_model = np.concatenate(pattern_segments)
        observed_prices = price_data[self.start_i:self.break_i]
        average_price = np.mean(observed_prices)

        sum_squared_residuals = np.sum((observed_prices - pattern_prices_model) ** 2.0)
        sum_squared_total = np.sum((observed_prices - average_price) ** 2.0)

        r_squared: float = 1.0 - sum_squared_residuals / sum_squared_total
        self.pattern_r2 = r_squared


def check_hs(extrema_indices: list[int], data: np.array, i: int, early_find: bool, invert: bool) -> HSPattern | None:
    """
    Checks if the given extrema indices represent a valid Head and Shoulders pattern.

    :param extrema_indices: The indices of the local extrema in the data.
    :param data: np.array of price data.
    :param i: Position of the current price in the np.array of price data.
    :param early_find: Whether to detect patterns early before confirmation by price breaking the neckline.
    :param invert: Inverted or normal head and shoulders pattern.
    :return: None if the pattern is invalid, otherwise a HSPattern instance representing the detected pattern.
    """
    l_shoulder, l_armpit, head, r_armpit = extrema_indices

    if i - r_armpit < 2:
        return None

    # Find right shoulder as extreme price since r_armpit based on pattern type
    extreme_func = np.argmax if not invert else np.argmin
    r_shoulder = r_armpit + extreme_func(data[r_armpit + 1: i]) + 1

    # Head must be extreme compared to shoulders
    if (not invert and data[head] <= max(data[l_shoulder], data[r_shoulder])) or \
            (invert and data[head] >= min(data[l_shoulder], data[r_shoulder])):
        return None

    # Balance rule. Shoulders are extreme compared to the others' midpoint.
    r_midpoint = 0.5 * (data[r_shoulder] + data[r_armpit])
    l_midpoint = 0.5 * (data[l_shoulder] + data[l_armpit])
    if (not invert and (data[l_shoulder] < r_midpoint or data[r_shoulder] < l_midpoint)) or \
            (invert and (data[l_shoulder] > r_midpoint or data[r_shoulder] > l_midpoint)):
        return None

    # Symmetry rule. Time from shoulder to head are comparable
    r_to_h_time = r_shoulder - head
    l_to_h_time = head - l_shoulder
    if r_to_h_time > 2.5 * l_to_h_time or l_to_h_time > 2.5 * r_to_h_time:
        return None

    # Compute neckline
    neck_run = r_armpit - l_armpit
    neck_rise = data[r_armpit] - data[l_armpit]
    neck_slope = neck_rise / neck_run

    # neckline value at current index
    neck_val = data[l_armpit] + (i - l_armpit) * neck_slope

    # Confirm pattern when price is halfway from right shoulder
    if early_find:
        if (not invert and data[i] > r_midpoint) or (invert and data[i] < r_midpoint):
            return None
    else:
        # Price has yet to break neckline, unconfirmed
        if (not invert and data[i] > neck_val) or (invert and data[i] < neck_val):
            return None

    # Find beginning of pattern. Neck to left shoulder
    head_width = r_armpit - l_armpit
    pat_start = neck_start = -1
    for j in range(1, head_width):
        neck = data[l_armpit] + (l_shoulder - l_armpit - j) * neck_slope
        if l_shoulder - j < 0:
            return None

        if (not invert and data[l_shoulder - j] < neck) or (invert and data[l_shoulder - j] > neck):
            pat_start = l_shoulder - j
            neck_start = neck
            break

    if pat_start == -1:
        return None

    # Pattern confirmed if here :)
    pattern = HSPattern(inverted=invert)

    pattern.l_shoulder = l_shoulder
    pattern.r_shoulder = r_shoulder
    pattern.l_armpit = l_armpit
    pattern.r_armpit = r_armpit
    pattern.head = head

    pattern.l_shoulder_price = data[l_shoulder]
    pattern.r_shoulder_price = data[r_shoulder]
    pattern.l_armpit_price = data[l_armpit]
    pattern.r_armpit_price = data[r_armpit]
    pattern.head_price = data[head]

    pattern.start_i = pat_start
    pattern.break_i = i
    pattern.break_price = data[i]

    pattern.neck_start = neck_start
    pattern.neck_end = neck_val

    pattern.neck_slope = neck_slope
    pattern.head_width = head_width

    if not invert:
        pattern.head_height = data[head] - (data[l_armpit] + (head - l_armpit) * neck_slope)
    else:
        pattern.head_height = (data[l_armpit] + (head - l_armpit) * neck_slope) - data[head]

    pattern.compute_r2(data)

    # Experiemented with r-squared as a filter for H&S, but this can delay recognition.
    # It didn't seem terribly potent, may be useful as a filter in conjunction with other attributes
    # if one wanted to add a machine learning layer before trading these patterns.

    # if pat.pattern_r2 < 0.0:
    #    return None

    return pattern


def find_patterns(data: np.array, order: int = 6, early_find: bool = False):
    """
    Identifies potential Head and Shoulders (regular and inverted) patterns in the given data.

    :param data: The price data as a NumPy array.
    :param order: Used by the rolling window function to find local minima and maxima. Lower = more sensitive.
    Defaults to 6.
    :param early_find: Whether to detect patterns early before confirmation by price breaking the neckline. Setting to
    False means waiting until the pattern is fully formed to detect it, but can result in missing the opportunity to
    get in.
    :return: A tuple containing lists of identified regular and inverted Head and Shoulders patterns.
    """
    assert (order >= 1), "Order must be at least 1."

    last_is_top = False
    recent_extrema = deque(maxlen=5)
    recent_types = deque(maxlen=5)  # -1 for bottoms 1 for tops

    # Lock variables to prevent finding the same pattern multiple times
    hs_lock = False
    ihs_lock = False

    ihs_patterns = []  # Inverted (bullish)
    hs_patterns = []  # Regular (bearish)
    for i in range(len(data)):

        if rw_top(data, i, order):
            recent_extrema.append(i - order)
            recent_types.append(1)
            ihs_lock = False
            last_is_top = True

        if rw_bottom(data, i, order):
            recent_extrema.append(i - order)
            recent_types.append(-1)
            hs_lock = False
            last_is_top = False

        if len(recent_extrema) < 5:
            continue

        hs_alternating = True
        ihs_alternating = True

        if last_is_top:
            for j in range(2, 5):
                if recent_types[j] == recent_types[j - 1]:
                    ihs_alternating = False

            for j in range(1, 4):
                if recent_types[j] == recent_types[j - 1]:
                    hs_alternating = False

            ihs_extrema = list(recent_extrema)[1:5]
            hs_extrema = list(recent_extrema)[0:4]
        else:

            for j in range(2, 5):
                if recent_types[j] == recent_types[j - 1]:
                    hs_alternating = False

            for j in range(1, 4):
                if recent_types[j] == recent_types[j - 1]:
                    ihs_alternating = False

            ihs_extrema = list(recent_extrema)[0:4]
            hs_extrema = list(recent_extrema)[1:5]

        def check_lock(lock, alternating, extrema, invert) -> HSPattern | None:
            if lock or not alternating:
                return None
            return check_hs(extrema, data, i, early_find, invert=invert)

        ihs_pattern = check_lock(ihs_lock, ihs_alternating, ihs_extrema, True)
        hs_pattern = check_lock(hs_lock, hs_alternating, hs_extrema, False)

        if hs_pattern is not None:
            hs_lock = True
            hs_patterns.append(hs_pattern)

        if ihs_pattern is not None:
            ihs_lock = True
            ihs_patterns.append(ihs_pattern)

    return hs_patterns, ihs_patterns


def plot_hs(data: pd.DataFrame, pattern: HSPattern, padding: int, filepath: str = None):
    """
    Plots the Head and Shoulders pattern with the provided data and padding.

    :param data: Data to plot.
    :param pattern: Pattern details to highlight in the plot.
    :param padding: Number of periods beyond the pattern to include in the plot for additional context.
    :param filepath: The file path to save the plot to.
    """
    padding = max(padding, 0)

    idx = data.index
    chart_start = max(pattern.start_i - padding, 0)
    chart_end = min(pattern.break_i + 1 + padding, len(data))
    data = data.iloc[chart_start:chart_end]

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figsize to ensure adequate space for labels
    datetime_format = '%m-%d'  # Just month and day for x-axis labels

    # Define the pattern lines to be drawn on the plot
    pattern_lines = [
        [(idx[pattern.start_i], pattern.neck_start), (idx[pattern.l_shoulder], pattern.l_shoulder_price)],
        [(idx[pattern.l_shoulder], pattern.l_shoulder_price), (idx[pattern.l_armpit], pattern.l_armpit_price)],
        [(idx[pattern.l_armpit], pattern.l_armpit_price), (idx[pattern.head], pattern.head_price)],
        [(idx[pattern.head], pattern.head_price), (idx[pattern.r_armpit], pattern.r_armpit_price)],
        [(idx[pattern.r_armpit], pattern.r_armpit_price), (idx[pattern.r_shoulder], pattern.r_shoulder_price)],
        [(idx[pattern.r_shoulder], pattern.r_shoulder_price), (idx[pattern.break_i], pattern.break_price)],
        [(idx[pattern.start_i], pattern.neck_start), (idx[pattern.break_i], pattern.neck_end)]
    ]

    ax.grid(True, color='#4e5294', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_title(
        f"{'Inverted ' * pattern.inverted}Head and Shoulders Pattern {idx[pattern.start_i].strftime('%Y-%m-%d %H:%M')}",
        fontsize=16)

    mpf.plot(
        data,
        type='candle',
        style='charles',  # Green and red candles
        ax=ax,
        datetime_format=datetime_format,
        alines=dict(
            alines=pattern_lines,
            colors=['w', 'w', 'w', 'w', 'w', 'w', 'r'],  # Colors for lines, 'r' highlights the neckline
            linewidths=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 1]
        ),
        volume="volume" in [col.lower() for col in data.columns],
        show_nontrading=True,
        tight_layout=True,
        warn_too_much_data=len(data) + 1,
    )

    if filepath:
        fig = plt.gcf()
        fig.savefig(filepath)

    plt.show()


def main():
    data = pd.read_csv('../../data/BTCUSDT3600.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')
    data = np.log(data)
    dat_slice = data['close'].to_numpy()

    padding = 10
    order = 15
    filepath = 'hs_pattern'

    hs_patterns, ihs_patterns = find_patterns(dat_slice, order, early_find=True)

    for i in range(len(hs_patterns)):
        plot_hs(data, hs_patterns[i], padding=padding, filepath=filepath + ".png")
        plot_hs(data, ihs_patterns[i], padding=padding, filepath=filepath + "_inverted.png")


if __name__ == '__main__':
    main()
