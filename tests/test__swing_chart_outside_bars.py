import pandas as pd

from technical_analysis_automation.swing_chart_candle_plotter import (
    BarType,
    _parse_data_set,
    detect_swing_extremes_across_data_set,
)


class TestSwingChartOutsideBars:
    def test__detect_swing_extremes_across_data_set__1bar(self) -> None:
        # Outside-bar should have 2 swing extremes at the top first then bottom.
        """
                |
            _|  |
          |  |  |
        |    |_
             |
        """

        data = {
            'date': pd.date_range(start='2023-01-01', periods=4, freq='D'),
            'high': [2.2, 3.5, 4, 5],
            'low': [1, 2, 0, 0.5],
            'open': [1.1, 2, 3, 1],
            'close': [2, 3, 1, 4.5],
            'volume': [100, 200, 300, 400]
        }
        df = pd.DataFrame(data)

        # Parse the data set
        data_set_bars = _parse_data_set(df)

        # Call the function
        result = detect_swing_extremes_across_data_set(data_set_bars, time_radius=1)

        # Expected swing extremes
        expected_data = {
            'index': [2, 2],
            'date': [pd.Timestamp('2023-01-03'),
                     pd.Timestamp('2023-01-03')],
            'high': [4, 4],
            'low': [0, 0],
            'open': [3, 3],
            'close': [1, 1],
            'volume': [300, 300],
            'bar_type': [BarType.OUTSIDE_BAR, BarType.OUTSIDE_BAR],
            'to_plot': ['high', 'low']
        }
        expected_df = pd.DataFrame(expected_data)

        print("\nActual Results:")
        print(result)
        print("\nExpected Results:")
        print(expected_df)

        # Assert the results
        pd.testing.assert_frame_equal(result, expected_df)
