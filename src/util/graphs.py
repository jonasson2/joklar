# IMPORTS
import itertools, numpy as np, matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('MacOSX')
AVG_CHAR_WIDTH = 0.55
#plt.ion()

plt.rc('figure.constrained_layout', use = True)
plt.rc('axes', grid = True)
for tick in ('xtick', 'ytick'):
    plt.rc(tick, labelsize=8)

def dimensions(measures: dict, maxrows, maxcols):
    # Return:
    #   row_measures = range of measures to be spread over rows
    #   col_measures = range of measures to be spread over columns
    # of each (sub)figure
    lens = [len(v) for v in measures.values()] + [100]
    n = len(measures)
    rows = 1
    cols = 1
    krow = 0
    while rows*lens[krow] <= maxrows:
        rows *= lens[krow]
        krow += 1
    kcol = krow
    while cols*lens[kcol] <= maxcols:
        cols *= lens[kcol]
        kcol += 1
    return range(krow), range(krow, kcol), kcol, range(kcol+1, n), rows, cols

def w_fontsize(text, ax, max_fontsize: int = 10) -> float:
    # Estimate the font size based on the length of the text and the available width
    pts_per_pix = 72/ax.figure.dpi
    width_pts = ax.get_position().width * ax.figure.bbox.width * pts_per_pix
    fontsize = width_pts / (len(text) * AVG_CHAR_WIDTH)
    return min(max_fontsize, fontsize)

def h_fontsize(text, ax, max_fontsize: int = 10) -> float:
    # Estimate the font size based on the length of the text and the available height
    pts_per_pix = 72/ax.figure.dpi
    height_pts = ax.get_position().height * ax.figure.bbox.height * pts_per_pix
    fontsize = height_pts / (len(text) * AVG_CHAR_WIDTH)
    return min(max_fontsize, fontsize)

def subplots(fig, m, n):
    ax = np.atleast_2d(fig.subplots(m, n, sharex=True, sharey=True))
    if n == 1:
        ax = ax.T
    return ax

def plots(parameters: dict, maxrows, maxcols):
    row_meas, col_meas, subfig_meas, fig_meas, rows, cols = dimensions(parameters, maxrows, maxcols)
    keys = list(parameters.keys())
    values = list(parameters.values())
    row_keys = [keys[k] for k in row_meas]
    col_keys = [keys[k] for k in col_meas]
    subfig_key = keys[subfig_meas]
    fig_keys = [keys[k] for k in fig_meas]
    ax_dict = {}                                    
    for fig_values in itertools.product(*(parameters[key] for key in fig_keys)):
        fig = plt.figure(figsize=(18, 8))
        header = ", ".join(f"{k}={c}" for k, c in zip(fig_keys, fig_values))
        fig.suptitle(header)
        nsubfig = len(parameters[subfig_key])
        subfigs = fig.subfigures(1, nsubfig)
        for subfig, subfig_value in zip(subfigs, parameters[subfig_key]):
            subfig.suptitle(f"{subfig_key} = {subfig_value}")
        for subfig, subfig_value in zip(subfigs, parameters[subfig_key]):
            subfig.suptitle(f"{subfig_key} = {subfig_value}")
            ax = subplots(subfig, rows, cols)
            for c, col_val in enumerate(itertools.product(*(values[k] for k in col_meas))):
                col_header = ", ".join(f"{k}={v}" for k, v in zip(col_keys, col_val))
                fontsize = w_fontsize(col_header, ax[0,0])
                ax[0, c].set_title(col_header, fontsize=fontsize)
            for r, row_val in enumerate(itertools.product(*(values[k] for k in row_meas))):
                row_header = ", ".join(f"{k}={v}" for k, v in zip(row_keys, row_val))
                fontsize = h_fontsize(row_header, ax[r,0])
                ax[r, 0].set_ylabel(row_header, fontsize=fontsize)
                for c, col_val in enumerate(itertools.product(*(values[k] for k in col_meas))):
                    val = list(row_val) + list(col_val) + [subfig_value] + list(fig_values)
                    ax_dict[tuple(val)] = ax[r, c]
    return ax_dict

if __name__ == "__main__":
    measures = {"A":[1,2], "B":[4,5], "long-C-title":[7,8], "long-D-title":[10,11], "E":[13, 14], "F":[16, 17], "G":[18, 19]}
    row_meas, col_meas, subfig_meas, fig_meas, rows, cols = dimensions(measures, 10, 10)
    print(row_meas, col_meas, subfig_meas, fig_meas, rows, cols)
    ax_dict = plots(measures, 6, 6)
    print(ax_dict)
    plt.show()
    #wait for keypress
    input("Press Enter to continue...")
