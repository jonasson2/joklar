#!/usr/bin/env python
import socket, json

def test_client(host='elja-irhpc', port=52981):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        request_message = "Request task"
        s.sendall(request_message.encode())
        response = s.recv(1024).decode()
        params = json.loads(response)
        print("Received parameters from the server:")
        print(params)

if __name__ == "__main__":
    test_client()  # Call the test client function
