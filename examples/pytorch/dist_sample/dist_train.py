import dgl
import dgl.backend as F
from dgl.contrib.dist_graph_store import DistGraphStore, DistGraphStoreServer
import numpy as np
import ip_config
import argparse
import torch as th
import time
import numpy as np

parser = argparse.ArgumentParser(description='Partition a graph')
parser.add_argument('-p', '--partition_book', required=True, type=str,
                    help='The path of the partition book')
args = parser.parse_args()



dist_g = DistGraphStore(ip_config.server_namebook, args.partition_book)
dist_g.connect()

# num_nodes = 232965
num_nodes = 40000
shuffled = np.arange(num_nodes)
np.random.shuffle(shuffled)
seed_num = 1024
fanout = 10
steps = 2

duration = 30
n = num_nodes // seed_num
start_t = time.time()
e_tot = 0

i = 0
cnt = 0
t_wo_first = 0
e_wo_first = 0
while time.time() - start_t < duration:
  seed = F.tensor(shuffled[i * seed_num : (i+1) * seed_num])
  result = seed
  start = time.time()
  for layer in range(steps):
    result = th.unique(result)
    subg = dist_g.neighbor_sample(result, fanout)
    result = subg.edges()[0]
    e_tot += len(result)
    # if i != 0:
      # e_wo_first += len(result)
  # if i != 0:
    # t_wo_first += time.time() - start
  cnt += 1
  i = (i + 1) % n 
print(cnt, " | avg subgraph size:", e_tot / cnt)
# print(e_wo_first/t_wo_first)
print(time.asctime(time.localtime(time.time())))
print(e_tot / (time.time() - start_t))
  
# seed0 = random.sample(range(232965), seed_num)
# seed = [x for x in seed0 if dist_g.part_id(x) in part_id]


# nodes_per_part = np.zeros([dist_g.num_parts()])
# for nid in seed:
#   part_id = dist_g.part_id(nid)
#   print(nid, part_id)
#   nodes_per_part[part_id] += 1
# print(nodes_per_part)


# nodes_per_part = np.zeros([dist_g.num_parts()])
# for src in result.edges()[0]:
#   nodes_per_part[dist_g.part_id(src)] += 1
# print(nodes_per_part)

# dist_g.shut_down()
