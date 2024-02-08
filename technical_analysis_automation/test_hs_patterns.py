import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from head_shoulders import find_patterns, HSPattern


class PatternAnalysis:
    def __init__(self):
        self.patterns = []
        self.statistics = {
            'count': [],
            'avg': [],
            'wr': [],
            'total_ret': [],
            'avg_stop': [],
            'wr_stop': [],
            'total_ret_stop': []
        }

    def add_patterns(self, patterns, dat_slice, data_length, direction='regular'):
        df = convert_patterns_to_df(patterns, dat_slice, data_length, direction)
        self.patterns.append(df)

    def calculate_statistics(self):
        """Calculates and stores statistics for each pattern."""
        for df in self.patterns:
            if df.empty:
                self._append_empty_stats()
            else:
                self.statistics['count'].append(len(df))
                self.statistics['avg'].append(df['hold_return'].mean())
                self.statistics['wr'].append((df['hold_return'] > 0).mean())
                self.statistics['total_ret'].append(df['hold_return'].sum())
                self.statistics['avg_stop'].append(df['stop_return'].mean())
                self.statistics['wr_stop'].append((df['stop_return'] > 0).mean())
                self.statistics['total_ret_stop'].append(df['stop_return'].sum())

    def _append_empty_stats(self):
        """Appends NaN or 0 for statistics when no patterns are present."""
        for key in ['count', 'total_ret', 'total_ret_stop']:
            self.statistics[key].append(0)
        for key in ['avg', 'wr', 'avg_stop', 'wr_stop']:
            self.statistics[key].append(np.nan)


def get_pattern_return(data_: np.array, pattern: HSPattern, log_prices: bool = True) -> float:
    entry_price = pattern.break_price
    entry_i = pattern.break_i
    stop_price = pattern.r_shoulder_price

    if pattern.inverted:
        tp_price = pattern.neck_end + pattern.head_height
    else:
        tp_price = pattern.neck_end - pattern.head_height

    exit_price = -1
    for i in range(int(pattern.head_width)):
        if entry_i + i >= len(data_):
            return np.nan

        exit_price = data_[entry_i + i]
        if pattern.inverted and (exit_price > tp_price or exit_price < stop_price):
            break

        if not pattern.inverted and (exit_price < tp_price or exit_price > stop_price):
            break

    if pattern.inverted:  # Long
        if log_prices:
            return exit_price - entry_price
        else:
            return (exit_price - entry_price) / entry_price
    else:  # Short
        if log_prices:
            return entry_price - exit_price
        else:
            return -1 * (exit_price - entry_price) / entry_price


def convert_patterns_to_df(patterns, dat_slice, data_length, direction='regular'):
    df = pd.DataFrame()
    for i, pattern in enumerate(patterns):
        df.loc[i, 'head_width'] = pattern.head_width
        df.loc[i, 'head_height'] = pattern.head_height
        df.loc[i, 'r2'] = pattern.pattern_r2
        df.loc[i, 'neck_slope'] = pattern.neck_slope

        hp = int(pattern.head_width)
        if pattern.break_i + hp >= data_length:
            df.loc[i, 'hold_return'] = np.nan
        else:
            if direction == 'inverse':
                ret = dat_slice[pattern.break_i + hp] - dat_slice[pattern.break_i]
            else:  # Assume 'regular' direction if not specified as 'inverse'
                ret = -1 * (dat_slice[pattern.break_i + hp] - dat_slice[pattern.break_i])
            df.loc[i, 'hold_return'] = ret

        df.loc[i, 'stop_return'] = get_pattern_return(dat_slice, pattern)
    return df


def append_results_from_analysis(df, analysis, prefix):
    df[f'{prefix}_count'] = analysis.statistics['count']
    df[f'{prefix}_avg'] = analysis.statistics['avg']
    df[f'{prefix}_wr'] = analysis.statistics['wr']
    df[f'{prefix}_total'] = analysis.statistics['total_ret']
    df[f'{prefix}_avg_stop'] = analysis.statistics['avg_stop']
    df[f'{prefix}_wr_stop'] = analysis.statistics['wr_stop']
    df[f'{prefix}_total_stop'] = analysis.statistics['total_ret_stop']
    return df


def plot_performance(title_prefix, results_df, categories):
    """Plots performance metrics for given analysis categories."""
    for category in categories:
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"{title_prefix} Performance: {category.upper().replace('_', ' ')}", fontsize=20)

        # Choose metrics based on title prefix to differentiate between hold and stop rule performances
        if 'Stop Rule' in title_prefix:
            metrics = ['count', 'avg_stop', 'total_stop', 'wr_stop']
        else:
            metrics = ['count', 'avg', 'total', 'wr']

        metric_titles = ['Number of Patterns Found', 'Average Pattern Return', 'Sum of Returns', 'Win Rate']
        metric_ylabels = ['Number of Patterns', 'Average Log Return', 'Total Log Return', 'Win Rate Percentage']
        colors = ['blue', 'yellow', 'green', 'orange']

        for i, metric in enumerate(metrics):
            ax_pos = ax[i // 2, i % 2]
            col_name = f'{category}_{metric}'
            results_df[col_name].plot.bar(ax=ax_pos, color=colors[i])
            ax_pos.set_title(metric_titles[i])
            ax_pos.set_xlabel('Order Parameter')
            ax_pos.set_ylabel(metric_ylabels[i])

            if 'avg' in metric or 'total' in metric:
                ax_pos.hlines(0.0, xmin=-1, xmax=len(results_df), color='white')
            if 'wr' in metric:
                ax_pos.hlines(0.5, xmin=-1, xmax=len(results_df), color='white')

        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        plt.show()


def main():
    data = pd.read_csv('../data/BTCUSDT3600.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')

    data = np.log(data)
    dat_slice = data['close'].to_numpy()

    orders = list(range(1, 49))

    ihs_analysis = PatternAnalysis()
    hs_analysis = PatternAnalysis()
    ihs_early_analysis = PatternAnalysis()
    hs_early_analysis = PatternAnalysis()

    for order in orders:
        hs_patterns, ihs_patterns = find_patterns(dat_slice, order, False)
        hs_patterns_early, ihs_patterns_early = find_patterns(dat_slice, order, True)

        # Add patterns to analysis objects
        hs_analysis.add_patterns(hs_patterns, dat_slice, len(data))
        ihs_analysis.add_patterns(ihs_patterns, dat_slice, len(data), direction='inverse')
        hs_early_analysis.add_patterns(hs_patterns_early, dat_slice, len(data))
        ihs_early_analysis.add_patterns(ihs_patterns_early, dat_slice, len(data), direction='inverse')

    # Calculate statistics after all patterns are added
    hs_analysis.calculate_statistics()
    ihs_analysis.calculate_statistics()
    hs_early_analysis.calculate_statistics()
    ihs_early_analysis.calculate_statistics()

    results_df = pd.DataFrame(index=orders)

    results_df = append_results_from_analysis(results_df, ihs_analysis, 'ihs')
    results_df = append_results_from_analysis(results_df, ihs_early_analysis, 'ihs_early')

    results_df = append_results_from_analysis(results_df, hs_analysis, 'hs')
    results_df = append_results_from_analysis(results_df, hs_early_analysis, 'hs_early')

    plt.style.use('dark_background')

    hold_categories = ['ihs', 'hs', 'ihs_early', 'hs_early']
    stop_categories = ['ihs', 'hs', 'ihs_early', 'hs_early']

    plot_performance("Hold Period", results_df, hold_categories)
    plot_performance("Stop Rule", results_df, stop_categories)


if __name__ == '__main__':
    main()
