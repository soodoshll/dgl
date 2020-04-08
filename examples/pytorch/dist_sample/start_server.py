from dgl.contrib.dist_graph_store import DistGraphStoreServer
import ip_config
server = DistGraphStoreServer(ip_config.server_namebook, 0, 'ppi/part-0.dgl', 'ppi/part_book.npz', 1)
server.start()