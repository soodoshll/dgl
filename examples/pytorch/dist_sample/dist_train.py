import dgl
import dgl.backend as F
from dgl.contrib.dist_graph_store import DistGraphStore, DistGraphStoreServer
import numpy as np
import ip_config

# g_server = DistGraphStoreServer({}, "ppi/part_book.npz")


dist_g = DistGraphStore(ip_config.server_namebook, "ppi_single/part_book.npz")
dist_g.connect()
dist_g.neighbor_sample(F.tensor([0,1,2,3,4]), 2)
dist_g.shut_down()