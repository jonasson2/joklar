import numpy as np, pandas as pd, os, json
from par_util import get_paths

def get_taskfolder_path(taskfolder):
    runpath = get_paths()["run"]
    return f"{runpath}/{taskfolder}"

def combine_results(taskfolder):
    path = get_taskfolder_path(taskfolder)
    combined_parameters = {}
    combined_measures = {}
    #
    for task_folder in os.listdir(path):
        if task_folder.startswith("results-"):
            jsonfile = f"{path}/{task_folder}/measures.json"
            with open(jsonfile, 'r') as file:
                data = json.load(file)
                parameters, measures = data[0], data[1]
                #
                # Combine parameters
                for key, value in parameters.items():
                    if key in combined_parameters:
                        combined_parameters[key].append(value)
                    else:
                        combined_parameters[key] = [value]
                #
                # Combine measures
                for key, value in measures.items():
                    if key in combined_measures:
                        combined_measures[key].append(value)
                    else:
                        combined_measures[key] = [value]
    #

    return combined_parameters, combined_measures

# function to write the combined 

if __name__ == "__main__":
    taskfolder = "run1"
    (parameters, measures) = combine_results(taskfolder)
    print("Parameters:", parameters)
    print("Measures:", measures)
