import argparse
import yaml
import os
from itertools import product

experiments = {
    'output_dir': ['week3/experiments'],
    'max_train': [500, 1000],
    'classes': [('cello', 'guitar'),
                ('cello', 'guitar', 'english-horn'),
                ('cello', 'guitar', 'english-horn', 'tuba'),
                ('cello', 'guitar', 'english-horn', 'tuba', 'clarinet', 'french-horn')],
    'sr': [8000],
    'window_size': [90e-3],
    'preprocessor':
        {
            'name': ['vggish', 'ised_features'],
            'normalize': [False],
            'mfcc_kwargs': {
                'log_mels': [False],
                'n_mfcc': [13, 128]
            }
        },
    'model': {
        'weights': [True, False]
    },
    'pca': {
        'num_components': [2, 3]
    },
    'num_neighbors': [3, 5]
}

def gen_experiments(exps):
    k, v = zip(*exps.items())
    v = (gen_experiments(val) if isinstance(val, dict) else val for val in v)
    for bundle in product(*v):
        d = dict(zip(k, bundle))
        yield d



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_path', type=str,
                        default=os.path.join(os.getcwd(), 'experiments'))

    args = parser.parse_args()

    out_path = args.output_path

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    i = 0
    for idx, exp in enumerate(gen_experiments(experiments)):
        with open(os.path.join(out_path, f'exp_{idx}.yaml'), 'w') as outfile:
            exp['name'] = f'exp_{idx}'
            yaml.dump(exp, outfile, default_flow_style=False)
        i = idx
    print(f'generated {i} experiments!')