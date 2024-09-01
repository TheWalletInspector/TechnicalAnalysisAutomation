import numpy as np
import pandas as pd

from technical_analysis_automation.swing_chart_candle_plotter import (
    Bar,
    BarType,
    _is_local_extreme,
    _parse_data_set,
    detect_swing_extremes_across_data_set,
)


class TestSwingChartCandlePlotter:
    def test__is_data_point_a_local_swing__should_not_have_enough_data_points(self) -> None:
        data_set = [Bar(index=i, date=None, high=val, low=val, open=None, close=None, volume=None) for i, val in
                    enumerate([1, 2, 3, 4, 5, 4, 3, 2, 1])]
        result = _is_local_extreme(data_set_bars=data_set, current_index=1, time_radius=2, comparison_operator=np.max)
        assert result is False

    def test__is_data_point_a_local_swing__should_be_a_local_swing_top(self) -> None:
        data_set = [Bar(index=i, date=None, high=val, low=val, open=None, close=None, volume=None) for i, val in
                    enumerate([1, 2, 3, 4, 5, 4, 3, 2, 1])]
        result = _is_local_extreme(data_set_bars=data_set, current_index=4, time_radius=1, comparison_operator=np.max)
        assert result is True

    def test__is_data_point_a_local_swing__should_be_a_local_2d_wide_swing_top(self) -> None:
        data_set = [Bar(index=i, date=None, high=val, low=val, open=None, close=None, volume=None) for i, val in
                    enumerate([1, 2, 3, 4, 5, 4, 3, 2, 1])]
        result = _is_local_extreme(data_set_bars=data_set, current_index=4, time_radius=2, comparison_operator=np.max)
        assert result is True

    def test__is_data_point_a_local_swing__should_not_be_a_local_2d_wide_swing_top(self) -> None:
        data_set = [Bar(index=i, date=None, high=val, low=val, open=None, close=None, volume=None) for i, val in
                    enumerate([1, 2, 5, 4, 5, 4, 3, 2, 1])]
        result_first_max = _is_local_extreme(data_set_bars=data_set, current_index=2, time_radius=2,
                                             comparison_operator=np.max)
        result_second_max = _is_local_extreme(data_set_bars=data_set, current_index=4, time_radius=2,
                                              comparison_operator=np.max)
        assert result_first_max is False
        assert result_second_max is False

    def test__is_data_point_a_local_swing__should_not_be_a_local_swing_top(self) -> None:
        data_set = [Bar(index=i, date=None, high=val, low=val, open=None, close=None, volume=None) for i, val in
                    enumerate([1, 2, 5, 4, 5, 4, 3, 2, 1])]
        result = _is_local_extreme(data_set_bars=data_set, current_index=4, time_radius=2, comparison_operator=np.max)
        assert result is False

    def test__is_data_point_a_local_swing__should_be_a_local_swing_bottom(self) -> None:
        data_set = [Bar(index=i, date=None, high=val, low=val, open=None, close=None, volume=None) for i, val in
                    enumerate([1, 2, 3, 2, 5, 4, 3, 1, 2])]
        result = _is_local_extreme(data_set_bars=data_set, current_index=7, time_radius=1, comparison_operator=np.min)
        assert result is True

    def test__is_data_point_a_local_swing__should_not_be_a_2d_wide_local_swing_bottom(self) -> None:
        data_set = [Bar(index=i, date=None, high=val, low=val, open=None, close=None, volume=None) for i, val in
                    enumerate([4, 2, 3, 3, 5, 1, 3, 1, 2])]
        result_first_low = _is_local_extreme(data_set_bars=data_set, current_index=5, time_radius=2,
                                             comparison_operator=np.min)
        result_second_low = _is_local_extreme(data_set_bars=data_set, current_index=7, time_radius=2,
                                              comparison_operator=np.min)
        assert result_first_low is False
        assert result_second_low is False

    def test__detect_swing_extremes_across_data_set(self) -> None:
        # Create a sample DataFrame
        data = {
            'date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
            'high': [1, 2, 3, 4, 5, 4, 3, 2, 1, 2],
            'low': [0, 1, 2, 3, 2, 3, 2, 1, 0, 1],
            'open': [0.5, 1.5, 2.5, 3.5, 4.5, 3.5, 2.5, 1.5, 0.5, 1.5],
            'close': [0.8, 1.8, 2.8, 3.8, 4.8, 3.8, 2.8, 1.8, 0.8, 1.8],
            'volume': [100, 200, 300, 400, 500, 400, 300, 200, 100, 200]
        }
        df = pd.DataFrame(data)

        # Parse the data set
        data_set_bars = _parse_data_set(df)

        # Call the function
        result = detect_swing_extremes_across_data_set(data_set_bars, time_radius=2)

        # Expected swing extremes
        expected_data = {
            'index': [4, 4, 8, 8],
            'date': [pd.Timestamp('2023-01-05'),
                     pd.Timestamp('2023-01-05'),
                     pd.Timestamp('2023-01-09'),
                     pd.Timestamp('2023-01-09')],
            'high': [5, 5, 1, 1],
            'low': [4, 4, 0, 0],
            'open': [4.5, 4.5, 0.5, 0.5],
            'close': [4.8, 4.8, 0.8, 0.8],
            'volume': [500, 500, 100, 100],
            'bar_type': [BarType.OUTSIDE_BAR, BarType.OUTSIDE_BAR, BarType.DOWN_BAR, BarType.DOWN_BAR]
        }
        expected_df = pd.DataFrame(expected_data)

        print(result)
        print(expected_df)

        # Assert the results
        pd.testing.assert_frame_equal(result, expected_df)
