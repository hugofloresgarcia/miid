import argparse
import yaml
import os
from itertools import product
from labeler import utils
import pandas as pd


"""
generate a cartesian product of all possible configs
everything inside a list will be decomposed into subsets
"""
"""
MAKE SURE EVERYTHING (UNLESS ITS ANOTHER DICT) IS WRAPPED
IN A LIST BECAUSE IT WILL TRY TO ITERATE OVER EVERYTHING
"""
names = ['openl3']
input_reprs = ['mel128', 'mel256']
embedding_sizes = ['512', '6144']
content_types = ['env', 'music']
openl3_names = ['-'.join([name, input_repr, embedding_size, content_type])
            for name, input_repr, embedding_size, content_type
                in product(names, input_reprs, embedding_sizes, content_types)]

experiments = {
    'seed': list(range(50)),
    'max_train': [200],
    'classes': [('flute', 'french-horn')],
    'preprocessor': ['openl3-mel256-512-music'],
    'fischer_reweighting': [False],
    'pca_n_components': [None],
    'classifier': ['knn-3', 'knn-5', 'knn-7', 
                    'svm-rbf', 'svm-linear', 'svm-sigmoid', 
                    *[f'svm-poly-{degree}' for degree in range(1, 5)]],
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

        path = os.path.join(subdir)
        # path = change_name_if_exists(path)
        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path, name+'.yaml'), 'w') as outfile:
            conf['name'] = name
            conf['output_dir'] = path
            yaml.dump(conf, outfile, default_flow_style=False)

    print(f'generated {len(configs)} configs in {out_path}!')