import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np


def get_key(d, filter_list):
    key = ''
    for label in filter_list:
        key += str(d[label]) + '_'

    return key

def compare_trials(path_to_output, filter_by, metrics):

    assert os.path.exists(path_to_output)

    # load our csv
    data = pd.read_csv(
        os.path.join(path_to_output, 'output.csv')
    ).to_dict('records')

    # filter our data by preprocessor and weights
    conditions = {}


    for record in data:
        key = get_key(record, filter_by)
        if key not in conditions:
            conditions[key] = []
        else:
            conditions[key].append(record)

    num_trials = [len(l) for l in conditions.values()]
    num_trials = min(num_trials)

    # BOXPLOTS
    num_subplots = len(metrics)
    subplot_rows = int(np.sqrt(num_subplots))
    subplot_cols = num_subplots-subplot_rows

    # get a figure with out subplots
    fig = plt.figure(figsize=(16, 9))
    axes = fig.subplots(subplot_rows, subplot_cols)
    axes = [item for sublist in axes for item in sublist]

    fig.suptitle(f'metrics over {num_trials} trials')

    for ax, metric in zip(axes, metrics):
        # every value is a list of dict entries with output data for a single run
        # the input data for our boxplot
        x = [[d[metric] for d in value] for value in conditions.values()]

        # do a boxplot w our data
        ax.boxplot(x)
        ax.set_title(metric)

        ticks = [key for key in conditions.keys()]
        ax.set_xticks(range(1, len(ticks)+1))
        ax.set_xticklabels(ticks, fontsize='x-small', rotation=45)

    fig.tight_layout()
    fig.savefig(
        os.path.join(path_to_output, 'metrics.png')
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_output', '-p', type=str,
                        default='week3/experiments')

    parser.add_argument('--filter_by', '-f', nargs='+', type=str)

    parser.add_argument('--metrics', '-m', nargs='+', type=str,
                        default='metrics_accuracy_score')

    args = parser.parse_args()

    compare_trials(args.path_to_output, args.filter_by, args.metrics)


