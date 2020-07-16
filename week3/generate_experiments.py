import argparse
import yaml
import os
from itertools import product

"""
generate a cartesian product of all possible configs
everything inside a list will be decomposed into subsets
"""
"""
MAKE SURE EVERYTHING (UNLESS ITS ANOTHER DICT) IS WRAPPED
IN A LIST BECAUSE IT WILL TRY TO ITERATE OVER EVERYTHING
"""
experiments = {
    'max_train': [500],
    'classes': [('french-horn', 'english-horn'),
                ('french-horn', 'english-horn', 'tuba'),
                ('french-horn', 'english-horn', 'tuba', 'clarinet')],
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
        path += '_new'
        return change_name_if_exists(path)
    else:
        os.mkdir(path)
        return os.path.join(path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_path', '-o', type=str,
                        default=os.path.join(os.getcwd(), 'experiments'))

    args = parser.parse_args()

    out_path = args.output_path
    out_path = change_name_if_exists(out_path)
    i = 0
    for idx, exp in enumerate(gen_experiments(experiments)):
        name = f'exp_{idx}'
        path = os.path.join(out_path, name)
        path = change_name_if_exists(path)

        with open(os.path.join(path, name+'.yaml'), 'w') as outfile:
            exp['name'] = f'exp_{idx}'
            exp['output_dir'] = out_path
            yaml.dump(exp, outfile, default_flow_style=False)
        i = idx
    print(f'generated {i+1} experiments in {out_path}!')