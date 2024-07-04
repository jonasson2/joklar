#!/usr/bin/env python
# Server script that doles out tasks to workers. The task list is
# read from a csv file provided on the command line with format:
#

import socket, pandas as pd, json, sys, os, shutil
from par_util import get_server_host_and_port

if len(sys.argv) < 3:
    print("""
Server script that distributes tasks to client.py workers

USAGE
  server.py parameters.csv output-folder

DESCRIPTION
  The server should be run either on the local computer (for testing)
  or on the login-node of elja. The parameter file has format:
    
    id,param1,param2...
    0,val1,val2...
    1,val1,val2...
    ...  

  and it can be created using create_tasks.sh. The server distributes tasks to
  workers (client.py scripts) running on compute nodes or (for testing) on the
  local computer. If the task with id xx is allocated to a client the output
  will go to the files:

    ~/joklar/output-folder/results-xx/{measures.json,output.log}

""")
    sys.exit()

def clear_results(folder):
    if not os.path.exists(folder):
        return
    for item in os.listdir(folder):
        if item.startswith('results-'):
            folder_item = f"{folder}/{item}"
            if os.path.isdir(folder_item):
                shutil.rmtree(folder_item)
            else:
                os.remove(folder_item)

def run_server(param_file, folder, port=52981):
    tasks = pd.read_csv(param_file, index_col=0)
    tasks["done"] = False
    results = {}
    pending_indices = tasks.index.copy()
    ntask = len(tasks)
    print('pending_indices=\n', pending_indices)
    (host, port) = get_server_host_and_port()
    os.makedirs(folder, exist_ok = True)
    processes = set()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        print(f"Server waiting to deliver tasks on port {port}, host {host}")
        s.listen()
        while True:
            (conn, addr) = s.accept()
            with conn:
                request = conn.recv(1024).decode()
                if request.startswith("Request task:"):
                    process = request.split(":", 1)[1]
                    if pending_indices.empty:
                        conn.sendall("No more tasks".encode())
                        processes.discard(process)
                    else:
                        processes.add(process)
                        task_number = int(pending_indices[0])
                        print('task_number', task_number)
                        task = tasks.loc[task_number].to_dict()
                        params = [folder, task_number, task]
                        stri = json.dumps(params)
                        conn.sendall(json.dumps(params).encode())
                        pending_indices = pending_indices.drop(task_number)
                        print(f"Task {task_number}/{ntask} for folder {folder} delivered")
                elif request.startswith("Task finished:"):
                    task_nr = int(request.split(":", 1)[1])
                    print(f'Task {task_nr} finished')
                    tasks.loc[task_nr, 'done'] = True
                else:
                    sys.exit(f"Unknown request: {request}")
                print('tasks.done', tasks.done)
                print('len(processes)', len(processes))
                if all(tasks.done) and len(processes) == 0:
                    break
        print("All tasks finished")

if __name__ == "__main__":
    param_file = sys.argv[1]
    folder = sys.argv[2]
    clear_results(folder)
    run_server(param_file, folder)
