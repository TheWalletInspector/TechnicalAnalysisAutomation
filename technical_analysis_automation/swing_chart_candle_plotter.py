import logging
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import pandas as pd


@dataclass
class Bar:
    index: int
    date: pd.Timestamp | None
    high: float | None
    low: float | None
    open: float | None
    close: float | None
    volume: float | None


class bar_type(Enum):
    up_bar = auto()
    down_bar = auto()
    inside_bar = auto()
    outside_bar = auto()


def _detect_bar_type(current_bar: Bar, previous_bar: Bar) -> bar_type:
    if current_bar.high > previous_bar.high and current_bar.low > previous_bar.low:
        return bar_type.up_bar
    elif current_bar.high < previous_bar.high and current_bar.low < previous_bar.low:
        return bar_type.down_bar
    elif current_bar.high < previous_bar.high and current_bar.low > previous_bar.low:
        return bar_type.inside_bar
    elif current_bar.high > previous_bar.high and current_bar.low < previous_bar.low:
        return bar_type.outside_bar
    else:
        raise ValueError("Bar type not recognized")


def _build_comparison_function(bar_type: bar_type, current_bar: Bar) -> np.ufunc:
    if bar_type == bar_type.up_bar:
        return np.max
    elif bar_type == bar_type.down_bar:
        return np.min
    elif bar_type == bar_type.outisde_bar:
        if current_bar.open > current_bar.close:
            return np.max, np.min
        else:
            return np.min, np.max
    else:
        raise ValueError("Bar type not recognized")


def _is_local_extreme(data_set: np.array, current_index: int, time_radius: int, comparison_operator) -> bool:
    if current_index < time_radius:
        logging.info(f"current_index: {current_index} does not have enough data points to form a rolling window")
        return False

    start = max(0, current_index - time_radius)
    end = min(len(data_set), current_index + time_radius + 1)
    window = data_set[start:end]

    is_extreme = data_set[current_index] == comparison_operator(window) and np.sum(
        window == data_set[current_index]) == 1

    print(f"current_index: {current_index}")
    print(f"start: {start}")
    print(f"end: {end}")
    print(f"window: {window}")
    print(f"is_extreme: {is_extreme}")

    return bool(is_extreme)


def detect_swing_extremes_across_data_set(data_set: pd.DataFrame, time_radius: int) -> pd.DataFrame:
    """
    Detects swing highs and lows across the data set
    """
    swing_extremes = []
    for index, row in data_set.iterrows():
        current_bar = Bar(
            index=int(index),  # Using the correct index
            date=row.date,  # Assuming 'date' is the index after setting it
            high=row.high,
            low=row.low,
            open=row.open,
            close=row.close,
            volume=row.volume if 'volume' in row else None  # Handle volume if it exists
        )

        # Detect swing high
        comparison_operator = np.max if current_bar.open > current_bar.close else np.min
        comparison_field = 'high' if current_bar.open > current_bar.close else 'low'
        if _is_local_extreme(data_set.to_numpy(), current_bar.index, time_radius, comparison_operator):
            swing_extremes.append(current_bar)

        # Detect swing low
        # comparison_operator = np.min if current_bar.open < current_bar.close else np.max
        if _is_local_extreme(data_set.to_numpy(), current_bar.index, time_radius, comparison_operator):
            swing_extremes.append(current_bar)

    return pd.DataFrame([vars(extreme) for extreme in swing_extremes],
                        columns=["index", "date", "high", "low", "open", "close", "volume"])


def main() -> None:
    """
    Main function to detect swing extremes across the data set
    """
    # Load and prepare the data
    data = pd.read_csv('.././data/BTCUSDT86400.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    # data = data.set_index('date')

    # Detect swing extremes
    swing_extremes = detect_swing_extremes_across_data_set(data, time_radius=2)

    print("Swing Extremes")
    print(swing_extremes)

    # Uncomment and adjust the following block to plot the results if needed
    # import mplfinance as mpf
    # fig, axlist = mpf.plot(
    #     data,
    #     type='candle',  # candlestick chart
    #     style='yahoo',  # chart style
    #     addplot=[
    #         mpf.make_addplot(swing_extremes['high'], type='scatter', color='green', marker='^', markersize=100),
    #         mpf.make_addplot(swing_extremes['low'], type='scatter', color='red', marker='v', markersize=100),
    #     ],
    #     volume=False,
    #     figratio=(10, 6),  # figure size
    #     title='BTCUSDT Trading Chart',
    #     ylabel='Price',
    #     ylabel_lower='Volume'
    # )
    # plt.show()


if __name__ == '__main__':
    main()
