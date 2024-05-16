import os, socket, json, sys

def get_paths():
    import socket, os
    hostname = socket.gethostname()
    if hostname == 'makki':
        path = os.path.expanduser("~/drive/joklar/")
    elif hostname.startswith('elja') or hostname == 'pluto':
        path = os.path.expanduser("~/joklar/")
    else:
        s = f"hostname is {hostname}, it should be 'makki', 'pluto', or 'elja*'"
        raise EnvironmentError(s)
    return {
        'path': path,
        'run': path + 'parallel',
        'src': path + 'src',
        'data': path + 'data/lang'
    }

def get_server_host_and_port():
    login_node = os.getenv('SLURM_SUBMIT_HOST')
    port = 52981
    if login_node is None: # For testing, when not on compute node
        login_node = "127.0.0.1"
    return (login_node, port)

def get_param_from_server(process):
    (host, port) = get_server_host_and_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((host, port))
            message = f"Request task:{process}"
            s.sendall(message.encode())
            response = s.recv(1024).decode()
            return response
        except ConnectionRefusedError:
            print("Could not conect to server")
            sys.exit(1)

def send_finished_to_server(task_number):
    (host, port) = get_server_host_and_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        message = f"Task finished:{task_number}"
        s.sendall(message.encode())    
        
if __name__ == "__main__":
    print("Working directory path:", get_paths()["path"])
    params = get_param_from_server("<host>:<pid>")
    if params:
        print("Received parameters from the server:", params)
    else:
        print("No (more) parameters available")
        
