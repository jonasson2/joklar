#!/usr/bin/env python

from par_util import get_path

def write_parameters(path):
    from itertools import product
    parameters = {
        "modeltype":  ["unet"],
        "augment":    [False],
        "factor":     [0.1, 0.3],
        "patience":   [5, 10, 20],
        "min_lr":     [1e-5, 1e-6],
        "init_lr":    [1e-3, 1e-4],
        "batch_size": [8, 32, 128],
    }
    with open(path + "parameters.csv", "w") as f:
        print('Index', *parameters.keys(), sep=',', file=f)
        for (idx,combination) in enumerate(product(*parameters.values())):
            print(idx, *combination, sep=',', file=f)

PATH = get_path()
write_parameters(PATH)
