import logging
from dataclasses import dataclass
from enum import Enum, auto

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)


class BarType(Enum):
    UP_BAR = auto()
    DOWN_BAR = auto()
    INSIDE_BAR = auto()
    OUTSIDE_BAR = auto()


@dataclass
class Bar:
    index: int
    date: pd.Timestamp | None
    high: float | None
    low: float | None
    open: float | None
    close: float | None
    volume: float | None
    bar_type: BarType | None = None
    to_plot: str | None = None


def _detect_bar_type(current_bar: Bar, previous_bar: Bar) -> BarType:
    if current_bar.high > previous_bar.high and current_bar.low > previous_bar.low:
        return BarType.UP_BAR
    elif current_bar.high < previous_bar.high and current_bar.low < previous_bar.low:
        return BarType.DOWN_BAR
    elif current_bar.high < previous_bar.high and current_bar.low > previous_bar.low:
        return BarType.INSIDE_BAR
    elif current_bar.high > previous_bar.high and current_bar.low < previous_bar.low:
        return BarType.OUTSIDE_BAR
    else:
        raise ValueError("Bar type not recognized")


def _get_comparison_operators(bar_type: BarType, current_bar: Bar) -> tuple:
    if bar_type == BarType.UP_BAR:
        return (np.max,)
    elif bar_type == BarType.DOWN_BAR:
        return (np.min,)
    elif bar_type == BarType.OUTSIDE_BAR:
        return (np.max, np.min) if current_bar.open > current_bar.close else (np.min, np.max)
    else:
        raise ValueError(f"Bar type not recognized: {bar_type}")


def _is_local_extreme(data_set_bars: list[Bar], current_index: int, time_radius: int, comparison_operator) -> bool:
    if current_index < time_radius or current_index > len(data_set_bars) - time_radius - 1:
        logging.info(f"current_index: {current_index} does not have enough data points to form a rolling window")
        return False

    data_set_array = np.array([bar.high if comparison_operator == np.max else bar.low for bar in data_set_bars])

    # print("is_local_extreme")
    # print(f"comparison_operator: {comparison_operator.__name__}")
    # print(f"data_set_array: {data_set_array}")

    start = max(0, current_index - time_radius)
    end = min(len(data_set_array), current_index + time_radius + 1)
    window = data_set_array[start:end]

    is_extreme = data_set_array[current_index] == comparison_operator(window) and np.sum(
        window == data_set_array[current_index]) == 1

    return bool(is_extreme)


def detect_swing_extremes_across_data_set(data_set_bars: list[Bar], time_radius: int) -> pd.DataFrame:
    swing_extremes = []
    for index, current_bar in enumerate(data_set_bars):
        previous_bar = data_set_bars[index - 1] if index > 0 else None

        if previous_bar:
            current_bar.bar_type = _detect_bar_type(current_bar=current_bar, previous_bar=previous_bar)
            if current_bar.bar_type == BarType.INSIDE_BAR:
                continue

            comparison_operators = _get_comparison_operators(bar_type=current_bar.bar_type, current_bar=current_bar)

            for operator in comparison_operators:
                if _is_local_extreme(data_set_bars, current_bar.index, time_radius, operator):
                    to_plot_value = "high" if operator == np.max else "low"

                    new_bar = Bar(
                        index=current_bar.index,
                        date=current_bar.date,
                        high=current_bar.high,
                        low=current_bar.low,
                        open=current_bar.open,
                        close=current_bar.close,
                        volume=current_bar.volume,
                        bar_type=current_bar.bar_type,
                        to_plot=to_plot_value
                    )
                    swing_extremes.append(new_bar)

    return pd.DataFrame([vars(extreme) for extreme in swing_extremes],
                        columns=["index", "date", "high", "low", "open", "close", "volume", "bar_type", "to_plot"])


def plot_ohlc_with_swings(data: pd.DataFrame, swings: pd.DataFrame) -> None:
    # Plot the OHLC chart
    ohlc_data = data.set_index('date')
    mpf.plot(ohlc_data, type='ohlc', style='charles', title='OHLC Chart with Swings', ylabel='Price', volume=True)

    # Plot the swings
    fig, ax = plt.subplots()
    mpf.plot(ohlc_data, type='ohlc', style='charles', ax=ax, volume=False)

    for i in range(1, len(swings)):
        if swings.iloc[i - 1]['to_plot'] == 'high' and swings.iloc[i]['to_plot'] == 'low':
            ax.plot([swings.iloc[i - 1]['date'], swings.iloc[i]['date']],
                    [swings.iloc[i - 1]['high'], swings.iloc[i]['low']], 'r-')
        elif swings.iloc[i - 1]['to_plot'] == 'low' and swings.iloc[i]['to_plot'] == 'high':
            ax.plot([swings.iloc[i - 1]['date'], swings.iloc[i]['date']],
                    [swings.iloc[i - 1]['low'], swings.iloc[i]['high']], 'g-')

    plt.show()


def _parse_data_set(data: pd.DataFrame) -> list[Bar]:
    data_set_bars = []
    for index, row in data.iterrows():
        data_set_bars.append(Bar(
            index=int(index),
            date=row.date,
            high=row.high,
            low=row.low,
            open=row.open,
            close=row.close,
            volume=row.volume if 'volume' in row else None
        ))

    return data_set_bars


def main() -> None:
    data = pd.read_csv('.././data/BTCUSDT86400.csv')
    data['date'] = data['date'].astype('datetime64[s]')

    data_set_bars: list[Bar] = _parse_data_set(data)
    swing_extremes = detect_swing_extremes_across_data_set(data_set_bars, time_radius=2)

    plot_ohlc_with_swings(data, swing_extremes)


if __name__ == '__main__':
    main()
