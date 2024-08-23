import numpy as np

from technical_analysis_automation.swing_chart_candle_plotter import _is_local_extreme


class TestSwingChartCandlePlotter:
    def test__is_data_point_a_local_swing__should_not_have_enough_data_points(self) -> None:
        result = _is_local_extreme(data_set=np.array([1, 2, 3, 4, 5, 4, 3, 2, 1]),
                                   current_index=1,
                                   time_radius=2,
                                   is_top=True)
        assert result is False

    def test__is_data_point_a_local_swing__should_be_a_local_swing_top(self) -> None:
        result = _is_local_extreme(data_set=np.array([1, 2, 3, 4, 5, 4, 3, 2, 1]),
                                   current_index=4,
                                   time_radius=1,
                                   is_top=True)
        assert result is True

    def test__is_data_point_a_local_swing__should_be_a_local_2d_wide_swing_top(self) -> None:
        result = _is_local_extreme(data_set=np.array([1, 2, 3, 4, 5, 4, 3, 2, 1]),
                                   current_index=4,
                                   time_radius=2,
                                   is_top=True)
        assert result is True

    def test__is_data_point_a_local_swing__should_not_be_a_local_2d_wide_swing_top(self) -> None:
        data_set = np.array([1, 2, 5, 4, 5, 4, 3, 2, 1])

        result_first_max = _is_local_extreme(data_set=data_set,
                                             current_index=2,
                                             time_radius=2,
                                             is_top=True)

        result_second_mox = _is_local_extreme(data_set=data_set,
                                              current_index=4,
                                              time_radius=2,
                                              is_top=True)

        assert result_first_max is False
        assert result_second_mox is False

    def test__is_data_point_a_local_swing__should_not_be_a_local_swing_top(self) -> None:
        result = _is_local_extreme(data_set=np.array([1, 2, 5, 4, 5, 4, 3, 2, 1]),
                                   current_index=4,
                                   time_radius=2,
                                   is_top=True)
        assert result is False

    def test__is_data_point_a_local_swing__should_not_be_a_local_swing_bottom(self) -> None:
        result = _is_local_extreme(data_set=np.array([1, 2, 3, 2, 5, 4, 3, 1, 2]),
                                   current_index=7,
                                   time_radius=1,
                                   is_top=False)
        assert result is True

    def test__is_data_point_a_local_swing__should_be_a_local_swing_bottom(self) -> None:
        data_set = np.array([4, 2, 3, 3, 5, 1, 3, 1, 2])

        result_first_low = _is_local_extreme(data_set=data_set,
                                             current_index=5,
                                             time_radius=1,
                                             is_top=False)

        result_second_low = _is_local_extreme(data_set=data_set,
                                              current_index=7,
                                              time_radius=1,
                                              is_top=False)

        assert result_first_low is True
        assert result_second_low is True


    def test__is_data_point_a_local_swing__should_not_be_a_2d_wide_local_swing_bottom(self) -> None:
        data_set = np.array([4, 2, 3, 3, 5, 1, 3, 1, 2])

        result_first_low = _is_local_extreme(data_set=data_set,
                                             current_index=5,
                                             time_radius=2,
                                             is_top=False)

        result_second_low = _is_local_extreme(data_set=data_set,
                                              current_index=7,
                                              time_radius=2,
                                              is_top=False)

        assert result_first_low is False
        assert result_second_low is False
