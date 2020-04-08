import dgl
import dgl.backend as F
from dgl.contrib.dist_graph_store import DistGraphStore, DistGraphStoreServer
import numpy as np
import ip_config

# g_server = DistGraphStoreServer({}, "ppi/part_book.npz")


dist_g = DistGraphStore(ip_config.server_namebook, "ppi/part_book.npz")
dist_g.connect()
