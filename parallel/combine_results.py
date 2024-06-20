import numpy as np, pandas as pd, os, json
from par_util import get_paths
import matplotlib.pyplot as plt
import pprint

def pp(data1, data2=None):
    ppr = pprint.PrettyPrinter(width=100, compact=True).pprint
    if data2 is None:
        ppr(data1)
    else:
        print(data1)
        ppr(data2)

def get_taskfolder_path(taskfolder):
    runpath = get_paths()["run"]
    return f"{runpath}/{taskfolder}"

import os
import json

def combine_results(taskfolder):
    def move_singletons_to_end(par: dict) -> dict:
        singletons = {k: v for k, v in par.items() if len(v) == 1}
        non_singletons = {k: v for k, v in par.items() if len(v) > 1}
        combined = {**non_singletons, **singletons}
        return combined

    def sets_to_sorted_lists(par: dict) -> dict:
        par = {key: sorted(list(value)) for key, value in par.items()}
        return par

    path = get_taskfolder_path(taskfolder)
    measures = []
    configurations = []
    task_folders = sorted(
        [f for f in os.listdir(path) if f.startswith("results-")],
        key=lambda x: int(x.split('-')[1])
    )
    first_file = True
    for task_folder in task_folders:
        jsonfile = f"{path}/{task_folder}/measures.json"
        with open(jsonfile, 'r') as file:
            data = json.load(file)
            (param, meas) = data
            configurations.append(param)
            measures.append(meas)
            if first_file:
                parameters = {p: {param[p]} for p in param}
                first_file = False
            else:
                for p in param:
                    assert(p in parameters)
                    parameters[p].add(param[p])
    parameters = move_singletons_to_end(parameters)
    parameters = sets_to_sorted_lists(parameters)
    return parameters, configurations, measures

def create_val_loss_dict(parameters, configurations, measures):
    val_loss_dict = {}
    for c,m in zip(configurations, measures):
        val_loss_curve = m['val_loss_curve']
        param_key = tuple(c[k] for k in parameters.keys())
        val_loss_dict[param_key] = val_loss_curve
    return val_loss_dict

def plot_loss(parameters, val_loss_dict):
    from graphs import plots
    def last_row(ax):
        nrows, ncols, start, _ = ax.get_subplotspec().get_geometry()
        return start//ncols == nrows - 1
    axes = plots(parameters, 4, 4)
    def first_col(ax):
        _, ncols, start, _ = ax.get_subplotspec().get_geometry()
        return start % ncols == 0
    for key in val_loss_dict:
        ax = axes[key]
        loss = np.array(val_loss_dict[key][0])
        std = np.array(val_loss_dict[key][1])
        bracketp = loss + std
        bracketm = loss - std
        ax.plot(loss, color="darkblue")
        ax.plot(bracketp, color = "darkred", linestyle = "dashed")
        ax.plot(bracketm, color = "darkred", linestyle = "dashed")
        #ax.set_ylabel("Loss")
        ax.set_xlim(0, len(loss))
        if first_col(ax):
            ax.set_ylabel(ax.get_ylabel() + '\nLoss')
            pass
        if last_row(ax):
            ax.set_xlabel("Epoch")
    pass

if __name__ == "__main__":
    taskfolder = "run1"
    parameters, configurations, measures = combine_results(taskfolder)
    val_loss_dict = create_val_loss_dict(parameters, configurations, measures)
    pp("Parameters:", parameters)
    pp("Measures, first entry:", measures[0])
    pp("Val_loss_dict, first key:", next(iter(val_loss_dict.keys())))
    pp("Val_loss_dict, first value:", next(iter(val_loss_dict.values())))
    plot_loss(parameters, val_loss_dict)
    plt.show()
    input('Press Enter to continue...')
    pass

