#!/usr/bin/env python

# External imports
from time import sleep
import socket, os, sys, json

# Start logger
import log # local
log.start('x.log', toscreen=True)
log.info('Logger started')

# Local imports
from par_util import get_param_from_server, send_finished_to_server, get_paths
paths = get_paths()
sys.path.append(paths["src"])
from run_task import fit_model

# Client that runs tasks while they are available
def client():
    # Create folder for task and make it current

    process = f"{socket.gethostname()}:{os.getpid()}"
    runpath = paths["run"]
    datapath = paths["data"]
    
    while True:
        response_str = get_param_from_server(process)
        if response_str == 'No more tasks':
            print(f"No more tasks for process {process}")
            break
        else:
            response = json.loads(response_str)
            (folder, task_number, params) = response
            modelpath = f"{runpath}/{folder}/results-{task_number}"
            os.makedirs(modelpath, exist_ok=True)
            os.chdir(modelpath)
            log.start("output.log", toscreen=True)
            log.info(f"Running task {task_number} on {process} with "\
                     f"parameters:\n{params}")
            fit_model(task_number, params,   datapath)
            sleep(1)
            send_finished_to_server(task_number)

if __name__ == "__main__":
    client()
