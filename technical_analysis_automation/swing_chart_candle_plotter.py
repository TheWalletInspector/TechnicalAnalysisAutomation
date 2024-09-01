import logging
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)


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
    if current_index < time_radius or current_index > len(data_set_bars) - time_radius:
        logging.info(f"current_index: {current_index} does not have enough data points to form a rolling window")
        return False

    data_set_array = np.array([bar.high if comparison_operator == np.max else bar.low for bar in data_set_bars])
    start = max(0, current_index - time_radius)
    end = min(len(data_set_array), current_index + time_radius + 1)
    window = data_set_array[start:end]

    is_extreme = data_set_array[current_index] == comparison_operator(window) and np.sum(
        window == data_set_array[current_index]) == 1

    logging.debug(f"current_index: {current_index}")
    logging.debug(f"start: {start}")
    logging.debug(f"end: {end}")
    logging.debug(f"window: {window}")
    logging.debug(f"is_extreme: {is_extreme}")

    return bool(is_extreme)


def detect_swing_extremes_across_data_set(data_set_bars: list[Bar], time_radius: int) -> pd.DataFrame:  # noqa: D103
    swing_extremes = []
    for index, current_bar in enumerate(data_set_bars):
        previous_bar = data_set_bars[index - 1] if index > 0 else None

        if previous_bar:
            current_bar.bar_type = _detect_bar_type(current_bar=current_bar, previous_bar=previous_bar)
            if current_bar.bar_type == BarType.INSIDE_BAR:
                continue

            comparison_operators = _get_comparison_operators(bar_type=current_bar.bar_type, current_bar=current_bar)

            print(f"current_bar: {current_bar}")

            for operator in comparison_operators:
                if _is_local_extreme(data_set_bars, current_bar.index, time_radius, operator):
                    swing_extremes.append(current_bar)

    return pd.DataFrame([vars(extreme) for extreme in swing_extremes],
                        columns=["index", "date", "high", "low", "open", "close", "volume", "bar_type"])


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


def main() -> None:  # noqa: D103
    data = pd.read_csv('.././data/BTCUSDT86400.csv')
    data['date'] = data['date'].astype('datetime64[s]')

    data_set_bars: list[Bar] = _parse_data_set(data)
    swing_extremes = detect_swing_extremes_across_data_set(data_set_bars, time_radius=2)

    # logging.info("Swing Extremes")
    # logging.info(swing_extremes)


if __name__ == '__main__':
    main()
