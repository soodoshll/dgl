from dgl.contrib.dist_graph_store import DistGraphStoreServer
import argparse
import ip_config

parser = argparse.ArgumentParser(description='Partition a graph')
parser.add_argument('-i', '--server_id', required=True, type=int,
                    help='The server id')
parser.add_argument('-d', '--data', required=True, type=str,
                    help='The path of local data')
parser.add_argument('-p', '--partition_book', required=True, type=str,
                    help='The path of partition book')
parser.add_argument('-n', '--client_num', required=True, type=int,
                    help='The total number of clients')
args = parser.parse_args()

server = DistGraphStoreServer(ip_config.server_namebook, args.server_id,
  args.data, args.partition_book, args.client_num)
server.start()