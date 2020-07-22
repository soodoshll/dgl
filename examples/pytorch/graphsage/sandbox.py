from ogb.nodeproppred import DglNodePropPredDataset

import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import time
import argparse
from _thread import start_new_thread
from functools import wraps
from dgl.data import RedditDataset
import tqdm
import traceback

from load_graph import load_reddit, load_ogb, inductive_split


def run(pid, args, data):
    # Unpack data
    g, train_idx, valid_idx, test_idx  = data
    train_nid = th.LongTensor(train_idx[pid])
    print("#train nodes: {} total: {}".format(len(train_nid), g.number_of_nodes()))
    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.sampling.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    dataloader = dgl.sampling.NodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    
    g.ndata['cnt'] = th.zeros([graph.number_of_nodes()])
    size_input = []
    size_seed = []
    for epoch in range(args.num_epochs):
        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            visited = th.cat([input_nodes, seeds])
            visited = th.unique(visited)
            g.ndata['cnt'][visited] += 1
            size_input.append(input_nodes.shape[0])
            size_seed.append(input_nodes.shape[0])
        print("proc {} Epoch {} average input size {}, seed size {}".format(pid, epoch, np.mean(size_input), np.mean(size_seed)))
    
    node_nonzero = th.nonzero(graph.ndata['cnt'])
    # print("nodes vistied {}".format(len(node_nonzero) / g.number_of_nodes()))

def run1(args, data):
    g, train_idx, valid_idx, test_idx  = data
    train_nid = th.LongTensor(train_idx)
    g.ndata['cnt'] = th.zeros([graph.number_of_nodes()])
    g.ndata['cnt'][train_nid] = 1
    g.update_all(message_func=fn.copy_src(src='cnt', out='m'),
            reduce_func=fn.sum(msg='m',out='cnt1'))
    g.update_all(message_func=fn.copy_src(src='cnt1', out='m'),
            reduce_func=fn.sum(msg='m',out='cnt2'))
    g.ndata['f'] = g.ndata['cnt'] + g.ndata['cnt1'] + g.ndata['cnt2']
    print(th.nonzero(g.ndata['f']).shape[0] / g.number_of_nodes())

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training sampled frequency counting")
    argparser.add_argument('--dataset', type=str, default='reddit')
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-partition', type=int, default=4)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--num-workers', type=int, default=0,
        help="Number of sampling processes. Use 0 for no extra process.")
    args = argparser.parse_args()
    
    dataset = DglNodePropPredDataset(name = "ogbn-products")

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    graph, label = dataset[0] # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)

    train_mask = th.zeros([graph.number_of_nodes()])
    train_mask[train_idx] = 1

    part = dgl.transform.metis_partition(graph, args.num_partition, balance_ntypes=train_mask)
    graph = dgl.graph(graph.edges())
    data = graph, train_idx, valid_idx, test_idx
    # run1(args, data)

    train_mask_part = [train_mask[part[i].ndata[dgl.NID]] for i in range(args.num_partition)]
    train_idx_part = [th.nonzero(p) for p in train_mask_part]
    data = graph, train_idx_part, valid_idx, test_idx
    for i in range(args.num_partition):
        print(train_idx_part[i].shape[0])
        run(i, args, data)