import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from head_shoulders import find_hs_patterns, HSPattern


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
        for key in self.statistics:
            if 'count' == key or 'total_ret' == key or 'total_ret_stop' == key:
                self.statistics[key].append(0)
            else:
                self.statistics[key].append(np.nan)


def get_pattern_return(data: np.array, pat: HSPattern, log_prices: bool = True) -> float:
    entry_price = pat.break_p
    entry_i = pat.break_i
    stop_price = pat.r_shoulder_p

    if pat.inverted:
        tp_price = pat.neck_end + pat.head_height
    else:
        tp_price = pat.neck_end - pat.head_height

    exit_price = -1
    for i in range(pat.head_width):
        if entry_i + i >= len(data):
            return np.nan

        exit_price = data[entry_i + i]
        if pat.inverted and (exit_price > tp_price or exit_price < stop_price):
            break

        if not pat.inverted and (exit_price < tp_price or exit_price > stop_price):
            break

    if pat.inverted:  # Long
        if log_prices:
            return exit_price - entry_price
        else:
            return (exit_price - entry_price) / entry_price
    else:  # Short
        if log_prices:
            return entry_price - exit_price
        else:
            return -1 * (exit_price - entry_price) / entry_price


data = pd.read_csv('.././data/BTCUSDT3600.csv')
data['date'] = data['date'].astype('datetime64[s]')
data = data.set_index('date')

data = np.log(data)
dat_slice = data['close'].to_numpy()

orders = list(range(1, 49))


def convert_patterns_to_df(patterns, dat_slice, data_length, direction=None):
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


ihs_analysis = PatternAnalysis()
ihs_early_analysis = PatternAnalysis()
hs_analysis = PatternAnalysis()
hs_early_analysis = PatternAnalysis()

for order in orders:
    hs_patterns, ihs_patterns = find_hs_patterns(dat_slice, order, False)
    hs_patterns_early, ihs_patterns_early = find_hs_patterns(dat_slice, order, True)

    hs_df = convert_patterns_to_df(hs_patterns, dat_slice, len(data))
    ihs_df = convert_patterns_to_df(ihs_patterns, dat_slice, len(data), direction='inverse')
    hs_early_df = convert_patterns_to_df(hs_patterns_early, dat_slice, len(data))
    ihs_early_df = convert_patterns_to_df(ihs_patterns_early, dat_slice, len(data), direction='inverse')

    hs_analysis.patterns.append(hs_df)
    ihs_analysis.patterns.append(ihs_df)
    hs_early_analysis.patterns.append(hs_early_df)
    ihs_early_analysis.patterns.append(ihs_early_df)


ihs_analysis.calculate_statistics()
ihs_early_analysis.calculate_statistics()
hs_analysis.calculate_statistics()
hs_early_analysis.calculate_statistics()


def append_results_from_analysis(df, analysis, prefix):
    df[f'{prefix}_count'] = analysis.statistics['count']
    df[f'{prefix}_avg'] = analysis.statistics['avg']
    df[f'{prefix}_wr'] = analysis.statistics['wr']
    df[f'{prefix}_total'] = analysis.statistics['total_ret']
    df[f'{prefix}_avg_stop'] = analysis.statistics['avg_stop']
    df[f'{prefix}_wr_stop'] = analysis.statistics['wr_stop']
    df[f'{prefix}_total_stop'] = analysis.statistics['total_ret_stop']
    return df


results_df = pd.DataFrame(index=orders)

results_df = append_results_from_analysis(results_df, ihs_analysis, 'ihs')
results_df = append_results_from_analysis(results_df, ihs_early_analysis, 'ihs_early')
results_df = append_results_from_analysis(results_df, hs_analysis, 'hs')
results_df = append_results_from_analysis(results_df, hs_early_analysis, 'hs_early')

plt.style.use('dark_background')


def plot_performance(analysis_categories, title_prefix, results_df):
    """Plots performance metrics for given analysis categories."""
    for category in analysis_categories:
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"{category.upper()} {title_prefix} Performance", fontsize=20)

        metrics = ['count', 'avg', 'total', 'wr']
        metric_titles = ['Number of Patterns Found', 'Average Pattern Return',
                         'Sum of Returns', 'Win Rate']
        metric_ylabels = ['Number of Patterns', 'Average Log Return',
                          'Total Log Return', 'Win Rate Percentage']
        colors = ['blue', 'yellow', 'green', 'orange']

        for i, metric in enumerate(metrics):
            ax_pos = ax[i // 2, i % 2]
            results_df[f'{category}_{metric}'].plot.bar(ax=ax_pos, color=colors[i])
            ax_pos.set_title(metric_titles[i])
            ax_pos.set_xlabel('Order Parameter')
            ax_pos.set_ylabel(metric_ylabels[i])

            if metric in ['avg', 'total']:
                ax_pos.hlines(0.0, xmin=-1, xmax=len(results_df), color='white')
            if metric == 'wr':
                ax_pos.hlines(0.5, xmin=-1, xmax=len(results_df), color='white')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


categories = ['ihs', 'hs', 'ihs_early', 'hs_early']
plot_performance(categories, "", results_df)
plot_performance(categories, "Early", results_df)
