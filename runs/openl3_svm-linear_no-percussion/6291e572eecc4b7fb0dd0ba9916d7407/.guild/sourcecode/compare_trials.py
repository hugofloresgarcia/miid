import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import itertools

import collections

import plotly.express as px
import plotly

def flatten_dict(d, parent_key='', sep='_'):
    """
    took this from
    https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_key(d, filter_list):
    key = ''
    for label in filter_list:
        key += str(d[label]) + '_'

    return key

def filter_dict(records, filter_list):
    conditions = {}
    for record in records:
        key = get_key(record, filter_list)
        if key not in conditions:
            conditions[key] = []
        else:
            conditions[key].append(record)

    return conditions

def statistical_tests(df, filter_by, metrics, path_to_output):

    # get all of our data
    data = df.to_dict('records')

    # filter by condition
    conditions = filter_dict(data, filter_by)
    pairs = {}
    diff = {}
    for condition1, condition2 in itertools.permutations(conditions.keys(), 2):
        tests = {}
        ## doing this for fischer reweighing
        if not ('False_' in condition1) and ('True_' in condition2):
            continue
        if not condition1[:-6] == condition2[:-5]:
            continue
        for metric in metrics:

            cond1 = conditions[condition1]
            cond2 = conditions[condition2]

            cond1 = np.array([item[metric] for item in cond1])
            cond2 = np.array([item[metric] for item in cond2])

            diff[metric] = cond1-cond2

            t_stat, t_p = stats.ttest_rel(cond1, cond2)
            w_stat, w_p = stats.wilcoxon(cond1, cond2)

            tests[metric] = dict(
                t_test_stat=round(t_stat, 7),
                t_test_pval=round(t_p, 13),
                wilcoxon_stat=round(w_stat, 7),
                wilcoxon_pval=round(w_p, 13)
            )
        pairs[f'{condition1[:-6]}'] = tests

    metrics = {}
    for pair in pairs:
        for metric in pairs[pair]:
            if metric not in metrics:
                metrics[metric] = {}
            metrics[metric][pair] = pairs[pair][metric]

    # make a stat tests dir
    tests_dir = os.path.join(path_to_output, 'stat_tests')
    if not os.path.exists(tests_dir):
        os.mkdir(tests_dir)

    for key, value in metrics.items():
        df = pd.DataFrame(value).transpose()
        df.to_csv(
            os.path.join(tests_dir, f'{key}.csv')
        )
    return

def do_boxplots(df, filter_by, metrics, path_to_output):

    # merge this condition for the boxplots
    for idx, row in df.iterrows():
        row['preprocessor'] = row['preprocessor'] + str(row['fischer_reweighting'])

    # now, do boxplots

    for metric in metrics:
        fig = px.box(df, x=filter_by, y=metric, points='all', color="fischer_reweighting")
        fig.write_html(
            os.path.join(path_to_output, metric + '.html')
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_output', '-p', type=str,
                        default='week3/experiments')

    parser.add_argument('--filter_by', '-f', nargs='+', type=str)

    parser.add_argument('--metrics', '-m', nargs='+', type=str,
                        default='metrics_accuracy_score')

    parser.add_argument('-t', '--t_test', action='store_true')

    args = parser.parse_args()

    df = pd.read_csv(
        os.path.join(args.path_to_output, 'output.csv')
    )

 
    
    if args.t_test:
        statistical_tests(df, args.filter_by, args.metrics, args.path_to_output)
    else: 
        do_boxplots(df, args.filter_by, args.metrics, args.path_to_output)

    print(f'output written to {args.path_to_output}')