import dgl
import dgl.backend as F
from dgl.contrib.dist_graph_store import DistGraphStore, DistGraphStoreServer
import numpy as np
import ip_config
import argparse

parser = argparse.ArgumentParser(description='Partition a graph')
parser.add_argument('-p', '--partition_book', required=True, type=str,
                    help='The path of the partition book')
args = parser.parse_args()

import numpy as np


dist_g = DistGraphStore(ip_config.server_namebook, args.partition_book)
dist_g.connect()

part_id=[0,2]

seed_num = 1000
import random
seed0 = random.sample(range(44906), seed_num)
seed = [x for x in seed0 if dist_g.part_id(x) in part_id]


nodes_per_part = np.zeros([dist_g.num_parts()])
for nid in seed:
  part_id = dist_g.part_id(nid)
  nodes_per_part[part_id] += 1
print(nodes_per_part)

result = dist_g.neighbor_sample(F.tensor(seed), 2)

nodes_per_part = np.zeros([dist_g.num_parts()])
for src in result.edges()[0]:
  nodes_per_part[dist_g.part_id(src)] += 1
print(nodes_per_part)

dist_g.shut_down()
