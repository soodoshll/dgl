import dgl
import numpy as np
from dgl.data.utils import load_graphs

class DistGraphStoreServer(object):
    def __init__(self, server_namebook, server_id, local_data, partition_book):
        self.server_namebook = server_namebook
        self.local_part = load_graphs([local_data])[0][0]
        self.part_book = np.load(partition_book)

    def neighbor_sample(self, seed, fanout):
        pass

class DistGraphStore(object):
    def __init__(self, server_namebook, partition_book):
        self.part_book = np.load(partition_book)
    
    def neighbor_sample(self, seed, fanout):
        pass    