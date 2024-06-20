# IMPORTS
import itertools
import matplotlib.pyplot as plt, matplotlib.figure as mpf
MAXWIDHEI = 2
plt.rc('figure.constrained_layout', use = True)
for tick in ('xtick', 'ytick'):
    plt.rc(tick, labelsize=8)

def split_type(nrows, ncols):
    if nrows/9 < ncols/16 < MAXWIDHEI:
        return 'vertically'
    elif ncols/16 < nrows/9 < MAXWIDHEI:
        return 'horizontally'
    else:
        return 'full'
    
def head(measures: dict):
    return next(iter(measures.items()))

def tail(measures: dict):
    return dict(itertools.islice(measures.items(), 1, None))

def max_splits(measures):
    nmeasures = len(measures)
    nsplit = [None]*nmeasures
    nrows = 1
    ncols = 1
    direction = ""
    maxSplit = 0
    measure_values = list(measures.values())
    for index in range(nmeasures):
        values = measure_values[index]
        nsplit[index] = len(values)
        match split_type(nrows, ncols):
            case 'vertically':
                nrows *= nsplit
                direction += 'v'
            case 'horizontally':
                ncols *= nsplit
                direction += 'h'
            case 'full':
                break
        maxSplit += 1
    return maxSplit, nsplit, direction

def split_graphs(fig, measures, nsplit, direction):
    if len(nsplit) == 0:
        return
    name, values = head(measures)
    match direction[0]:
        case 'v':
            bfigs = fig.subfigures(nsplit[0], 1)
            for (sf, val) in zip(subfigs, values):
                title = f"{name}={val}"
                sf.suptitle(title, x=0, y=0.5, ha='right', va='center', rotation=90)
        case 'h':
            subfigs = fig.subfigures(1, nsplit[0])
            for (sf, val) in zip(subfigs, values):
                title = f"{name}={val}"
                sf.suptitle(title)
    for sf in subfigs:
        split_graphs(sf, tail(measures), nsplit[1:], direction[1:])

def create_figures(measures):
    # measures is a dictionary {measure: list-of-values, ...}
    maxSplit, nsplit, direction = max_splits(measures)
    split_measures = dict(itertools.islice(measures.items(), maxSplit))
    remaining_keys = list(measures.keys())[maxSplit:]
    remaining_values = list(measures.values())[maxSplit:]
    for vals in itertools.product(*remaining_values):
        heading = ', '.join([f'{m}={v}' for m, v in zip(remaining_keys, vals)])
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(heading)
        split_graphs(fig, split_measures, nsplit, direction)
        print(heading)

if __name__ == "__main__":
    def create_test_measures(L):
        d = {}
        for (i,n) in enumerate(L):
            d[f'measure{i}'] = list(range(n))
        return d
    def run_test():
        L1 = [3, 2, 4, 5]
        L2 = [4, 3, 2, 4, 3, 2]
        L3 = [6, 2, 2, 4, 3, 2, 2]
        for (L,n) in zip([L1, L2, L3], [4, 6, 6]):
            measures = create_test_measures(L)
            ms, dir = max_splits(measures)
            print(f'max_splits={ms}, dir={dir}')
            assert(ms == n)
            #create_figures(measures)

    # Test
    run_test()
