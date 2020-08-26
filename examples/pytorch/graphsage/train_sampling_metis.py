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
#from torch.utils.tensorboard import SummaryWriter       
#writer = SummaryWriter("~/tensorboard_log")

from load_graph import load_reddit, load_ogb, inductive_split

class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.number_of_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.number_of_nodes()),
                sampler,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=args.num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, inputs, labels, val_nid, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid])

def load_subtensor(g, seeds, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = g.ndata['features'][input_nodes].to(device)
    batch_labels = g.ndata['labels'][seeds].to(device)
    return batch_inputs, batch_labels

class NeighborSampler(object):
    def __init__(self, g, fanouts):
        self.g = g
        self.fanouts = fanouts

    def sample_frontiers(self, seeds):
        # print(seeds)
        seeds = np.concatenate(seeds)
        flow = []
        # seeds = th.tensor(seeds) #.flatten()
        for fanout in self.fanouts:
            frontier = dgl.sampling.sample_neighbors(g, seeds, fanout, replace=False)
            frontier = dgl.to_block(frontier, seeds)
            seeds = frontier.srcdata[dgl.NID]
            flow.insert(0, frontier)
        return flow[0].srcdata[dgl.NID], flow[-1].dstdata[dgl.NID], flow


#### Entry point
def run(args, device, data):
    # Unpack data
    in_feats, n_classes, train_g, val_g, test_g = data
    train_nid = th.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = th.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = th.nonzero(~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]

    # Use metis to partition training nodes
    num_shard = 48
    if num_shard > 0:
        train_subgraph = train_g.subgraph(train_nid)
        train_part = dgl.transform.metis_partition(train_subgraph, len(train_nid) // args.batch_size * num_shard)
        train_part = [train_part[p] for p in train_part]
        train_batches = [train_subgraph.ndata[dgl.NID][p.ndata[dgl.NID]].numpy() for p in train_part] 
    else:
        num_shard = args.batch_size
        train_batches = [th.tensor([x]) for x in train_nid]

    # Create PyTorch DataLoader for constructing blocks
    sampler = NeighborSampler(train_g, [int(fanout) for fanout in args.fan_out.split(',')])
    dataloader = DataLoader(dataset=train_batches,
                        batch_size=num_shard,
                        collate_fn=sampler.sample_frontiers,
                        shuffle=True,
                        num_workers=args.num_workers)


    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    avg = 0
    iter_tput = []
    n_nodes = []
    n_edges = []

    label = train_g.ndata['labels'][train_nid]
    tot_dist = th.bincount(label, minlength=n_classes).float()/label.shape[0]
   
    timestamp = time.strftime("%m%d-%H%M%S", time.localtime())
    loss_log = open("log/loss-" + str(num_shard) + '-' + timestamp, "w")
    acc_log = open("log/acc-" + str(num_shard) + '-' + timestamp, "w")
    tot_t = 0  
    for epoch in range(args.num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        tic_step = time.time()
        num_nodes = 0
        num_edges = 0
        label_tv = []
        most_prop = []
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(train_g, seeds, input_nodes, device)
            batch_tv = label_dist_tv(tot_dist, batch_labels, n_classes)
            label_tv.append(batch_tv)
            label_mode = th.mode(batch_labels).values
            most = th.sum(batch_labels == label_mode).float().cpu().numpy()/batch_labels.shape[0]
            most_prop.append(most)
            blocks = [block.int().to(device) for block in blocks]

            # To rebalance the training nodes using reweighting
            # batch_dist = th.bincount(batch_labels, minlength=n_classes) + 1 / (label.shape[0] + n_classes)
            # weight = tot_dist.cuda() / batch_dist

            #loss_fcn = nn.CrossEntropyLoss(weight=weight)
            loss_fcn = nn.CrossEntropyLoss()
            loss_fcn = loss_fcn.to(device)
            
            num_nodes += len(input_nodes)
            num_edges += blocks[0].number_of_edges()
            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MiB'.format(
                    epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))
                loss_log.write("{:.4f}\n".format(loss.item()))
                loss_log.flush()
            tic_step = time.time()
        toc = time.time()
        tot_t += toc - tic
        print('Epoch Time(s): {:.4f} #nodes: {} #edges: {} | tv of label distribution {:.4f} | proportion of the most class {:.4f}'.format(toc - tic, num_nodes, num_edges, np.mean(label_tv), np.mean(most_prop)))
        if epoch >= 5:
            avg += toc - tic
        if (epoch + 1) % args.eval_every == 0:
            # eval_acc = evaluate(model, val_g, val_g.ndata['features'], val_g.ndata['labels'], val_nid, args.batch_size, device)
            # print('Eval Acc {:.4f}'.format(eval_acc))
            test_acc = evaluate(model, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_nid, args.batch_size, device)
            print('Test Acc: {:.4f}'.format(test_acc))
            acc_log.write('{:.2f} {:.4f}\n'.format(tot_t, test_acc))
            acc_log.flush()

    print('Avg epoch time: {}'.format(avg / (epoch - 4)))

def label_dist_tv(tot_dist, batch_labels, n_classes):
    batch_labels = batch_labels.cpu()
    batch_dist = th.bincount(batch_labels, minlength=n_classes).float() / batch_labels.shape[0]
    return 0.5*th.sum(th.abs(tot_dist-batch_dist)).numpy()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=0,
        help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='reddit')
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=4,
        help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--inductive', action='store_true',
        help="Inductive learning setting")
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    if args.dataset == 'reddit':
        g, n_classes = load_reddit()
    elif args.dataset == 'ogb-product':
        g, n_classes = load_ogb('ogbn-products')
    else:
        raise Exception('unknown dataset')

    in_feats = g.ndata['features'].shape[1]

    if args.inductive:
        train_g, val_g, test_g = inductive_split(g)
    else:
        train_g = val_g = test_g = g

    train_g.create_format_()
    val_g.create_format_()
    test_g.create_format_()
    # Pack data
    data = in_feats, n_classes, train_g, val_g, test_g
    print("#nodes:", train_g.number_of_nodes(), "#edges:", train_g.number_of_edges(), "#train:", th.sum(train_g.ndata['train_mask']))
    run(args, device, data)
