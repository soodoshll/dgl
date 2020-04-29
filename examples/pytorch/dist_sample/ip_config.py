port = 4354
server_num = 4
server_namebook = {
    i : ['127.0.0.1', str(port+i)] for i in range(server_num)
}