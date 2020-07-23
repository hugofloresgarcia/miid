import argparse
import yaml
import os
from itertools import product
from ised import utils
import pandas as pd

# i wanna have
# 20 different trials
# 4 different models per trial (ised, vgg, weights, no weights)
# compare restuls per TRIAL (4 of these)
# how do I do this?
# get output per single configuration, including figures?
# put all these figures together?

# IDEA: add a "group by arg", stating that you would like to group these by seed
#

"""
generate a cartesian product of all possible configs
everything inside a list will be decomposed into subsets
"""
"""
MAKE SURE EVERYTHING (UNLESS ITS ANOTHER DICT) IS WRAPPED
IN A LIST BECAUSE IT WILL TRY TO ITERATE OVER EVERYTHING
"""
experiments = {
    'seed': list(range(50)),
    'max_train': [200],
    'classes': [('french-horn', 'english-horn')],
    'sr': [8000],
    'window_size': [90e-3],
    'preprocessor':
        {
            'name': ['vggish', 'ised_features'],
            'normalize': [False],
            'mfcc_kwargs': {
                'log_mels': [False],
                'n_mfcc': [13]
            }
        },
    'model': {
        'weights': [True, False]
    },
    'pca': {
        'num_components': [2]
    },
    'num_neighbors': [3]
}

def gen_experiments(exps):
    k, v = zip(*exps.items())
    v = (gen_experiments(val) if isinstance(val, dict) else val for val in v)
    for bundle in product(*v):
        d = dict(zip(k, bundle))
        yield d

def change_name_if_exists(path):
    if os.path.exists(path):
        path += utils.get_time()
        return change_name_if_exists(path)
    else:
        os.makedirs(path)
        return os.path.join(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_path', '-o', type=str,
                        default=os.path.join(os.getcwd(), 'experiments'))

    parser.add_argument('--group_by', '-g', type=str, nargs='*',
                        default=None)

    args = parser.parse_args()

    out_path = args.output_path
    out_path = change_name_if_exists(out_path)

    # put all configs in one big list
    configs = [exp for exp in gen_experiments(experiments)]

    for idx, conf in enumerate(configs):
        name = f'exp_{idx}'

        # group by the stuff specified in the args
        if args.group_by is not None:
            # if we have multiple things to group by, add a subdir one by one
            if isinstance(args.group_by, list):
                subdir = os.path.join(out_path) # our parent subdir
                for subd in args.group_by: # add a subdir
                    if isinstance(conf[subd], tuple): # if our conf contents are also a list, lets just state the length
                        tag = len(conf[subd])
                    else:
                        tag = conf[subd]
                    subdir = os.path.join(subdir, f'{subd}_{tag}')
            # if we just got one thing to group by, just one subdir is fine
            else:
                subdir = os.path.join(out_path, f'{args.group_by}_{conf[args.group_by]}')
        else:
            subdir = out_path

        path = os.path.join(subdir, name)
        path = change_name_if_exists(path)

        with open(os.path.join(path, name+'.yaml'), 'w') as outfile:
            conf['name'] = name
            conf['output_dir'] = path
            yaml.dump(conf, outfile, default_flow_style=False)

    print(f'generated {len(configs)} configs in {out_path}!')