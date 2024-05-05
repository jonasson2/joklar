#!/usr/bin/env python

# Server script that doles out tasks to workers. The task list is
# read from a csv file with format:
#
# id,param1,param2...
# 0,val1,val2...
# 1,val1,val2...

import socket, pandas as pd, json

def run_server(host='elja-irhpc', port=52981):
    tasks = pd.read_csv("tasks.txt", index_col=0)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))            
        s.listen()
        for task_number in tasks.index:
            params = json.dumps(tasks.loc[task_number].to_dict())
            (conn, addr) = s.accept()
            with conn:
                print(f"Connected by {addr}")
                request = conn.recv(1024).decode()
                assert(request == "Request task")
                conn.sendall(params.encode())

if __name__ == "__main__":
    run_server()
