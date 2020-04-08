"""
This is basicly same as the partition script in Da's pull request.
"""
import numpy as np
import argparse
import signal
import dgl
from dgl import backend as F
from dgl.data.utils import load_graphs, save_graphs

def main():
    parser = argparse.ArgumentParser(description='Partition a graph')
    parser.add_argument('--data', required=True, type=str,
                        help='The file path of the input graph in the DGL format.')
    parser.add_argument('-k', '--num-parts', required=True, type=int,
                        help='The number of partitions')
    parser.add_argument('-m', '--method', required=True, type=str,
                        help='The partitioning method: random, metis')
    parser.add_argument('-o', '--output', required=True, type=str,
                        help='The output directory of the partitioned results')
    args = parser.parse_args()
    data_path = args.data
    num_parts = args.num_parts
    method = args.method
    output = args.output

    glist, _ = load_graphs(data_path)
    g = glist[0]

    if num_parts == 1:
        server_parts = {0: g}
        node_parts = np.zeros(g.number_of_nodes())
        g.ndata['part_id'] = F.zeros((g.number_of_nodes()), F.int64, F.cpu())
        g.ndata[dgl.NID] = F.arange(0, g.number_of_nodes())
        g.edata[dgl.EID] = F.arange(0, g.number_of_edges())
    elif args.method == 'metis':
        node_parts = dgl.transform.metis_partition_assignment(g, num_parts)
        server_parts = dgl.transform.partition_graph_with_halo(g, node_parts, 1)
        for part_id in server_parts:
            part = server_parts[part_id]
            part.ndata['part_id'] = F.gather_row(node_parts, part.ndata[dgl.NID])
    elif args.method == 'random':
        node_parts = np.random.choice(num_parts, g.number_of_nodes())
        server_parts = dgl.transform.partition_graph_with_halo(g, node_parts, 1)
        for part_id in server_parts:
            part = server_parts[part_id]
            part.ndata['part_id'] = F.gather_row(node_parts, part.ndata[dgl.NID])
    else:
        raise Exception('unknown partitioning method: ' + args.method)

    global2local = np.zeros(g.number_of_nodes(), dtype=np.int64) - 1
    tot_num_inner_edges = 0
    for part_id in range(num_parts):
        part = server_parts[part_id]

        if num_parts > 1:
            num_inner_nodes = len(np.nonzero(F.asnumpy(part.ndata['inner_node']))[0])
            num_inner_edges = len(np.nonzero(F.asnumpy(part.edata['inner_edge']))[0])
            print('part {} has {} nodes and {} edges. {} nodes and {} edges are inside the partition'.format(
                part_id, part.number_of_nodes(), part.number_of_edges(),
                num_inner_nodes, num_inner_edges))
            tot_num_inner_edges += num_inner_edges
            
            # TODO: Features of halo nodes are unnecessary.
            part.copy_from_parent()

            # build global2local mapping
            inner_nodes = np.argwhere(F.asnumpy(part.ndata['inner_node']))
            global_id = F.asnumpy(part.ndata[dgl.NID][inner_nodes])
            global2local[global_id] = inner_nodes

        save_graphs(output + '/part-' + str(part_id) + '.dgl', [part])

    # To save partition book 
    assert np.all(global2local >= 0)
    np.savez(output+'/part_book.npz', num_parts = num_parts, part_id = node_parts, global2local = global2local)
    

    num_cuts = g.number_of_edges() - tot_num_inner_edges
    if num_parts == 1:
        num_cuts = 0
    print('there are {} edges in the graph and {} edge cuts for {} partitions.'.format(
        g.number_of_edges(), num_cuts, num_parts))

if __name__ == '__main__':
    main()
