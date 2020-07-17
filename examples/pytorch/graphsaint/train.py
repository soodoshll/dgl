import argparse, time, math
import numpy as np

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import dgl
from dgl import DGLGraph

from dgl.data import PPIDataset
from dgl.data import KarateClub
from dgl.data import RedditDataset

import dgl.function as fn

from load_graph import load_reddit

class RandomWalkSampler(object):
    def __init__(self, g, walk_length, 
                 run_profile=False, 
                 profile_epochs=1, 
                 profile_batch_size=10):
        self.g = g
        self.walk_length = walk_length
        self.profile_epochs = profile_epochs
        self.profile_batch_size = profile_batch_size
        if run_profile:
            self.profile(profile_batch_size)
    
    def sample_nodes(self, seeds):
        rw = dgl.sampling.random_walk(g, seeds, length=self.walk_length)[0]
        return th.unique(th.flatten(rw))
        # return seeds

    def sample_subgraph(self, seeds):
        nodes = self.sample_nodes(seeds)
        return self.g.subgraph(nodes)

    def profile(self, num_workers=0):
        print("Estimating normalization statistics...")
        loader = DataLoader(self.g.nodes(),
                            batch_size=self.profile_batch_size,
                            shuffle=True,
                            collate_fn=self.sample_subgraph,
                            num_workers=num_workers
                            )
        node_cnt = th.zeros([self.g.number_of_nodes()])
        edge_cnt = th.zeros([self.g.number_of_edges()])
        cnt_subg = 0
        for i in range(self.profile_epochs):
            for bid, subg in enumerate(loader):
                node_cnt[subg.ndata[dgl.NID]] += 1
                edge_cnt[subg.edata[dgl.EID]] += 1
                cnt_subg += 1
                if bid % 1000 == 0:
                    print(bid)
        cnt_dst = node_cnt[self.g.edges(form='uv')[1]]
        self.g.ndata['norm'] = node_cnt / cnt_subg
        self.g.edata['norm'] = edge_cnt / cnt_dst

class SAINTLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 norm=False,
                 bias=True):
        super(SAINTLayer, self).__init__()
        self.fc = nn.Linear(in_feats, out_feats, bias=bias)
        self.activation = activation
        self.norm = norm

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc.weight, gain=gain)

    def forward(self, graph, feat):
        with graph.local_scope():
            graph.ndata['h'] = feat
            if self.norm:
                graph.update_all(fn.u_mul_e('h', 'norm', 'm'), fn.sum('m', 'neigh'))
            else:
                graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))
            rst = graph.ndata['neigh']
            rst = self.fc(rst)
            if self.activation:
                rst = self.activation(rst)
            return rst

class SAINT(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 activation,
                 ):
        super(SAINT, self).__init__()
        self.layers = nn.ModuleList()
        if isinstance(n_hidden, int):
            n_hidden = [n_hidden]
        self.layers.append(SAINTLayer(in_feats, n_hidden[0], activation))
        for i in range(len(n_hidden) - 1):
            self.layers.append(SAINTLayer(n_hidden[i], n_hidden[i+1], activation))
        self.lin = nn.Linear(sum(n_hidden), n_classes)

    def forward(self, graph, feat):
        h = feat
        out = []
        for layer in self.layers:
            h = layer(graph, h)
            out.append(h)
        h = self.lin(th.cat(out, dim=-1))
        return h.log_softmax(dim=-1)

def load_subtensor(g, nodes, labeled_nodes, device):
    batch_inputs = g.ndata['feature'][nodes].to(device)
    batch_labels = g.ndata['label'][labeled_nodes].to(device)
    return batch_inputs, batch_labels

def run(args, device, data):
    train_mask, val_mask, in_feats, labels, n_classes, g = data
    train_nid = g.nodes()
    sampler = RandomWalkSampler(g, 2)
    loader = DataLoader(train_nid,
                    batch_size=args.batch_size,
                    shuffle=True,
                    collate_fn=sampler.sample_subgraph,
                    num_workers=args.num_workers,
                    drop_last=True
                    )
    model = SAINT(in_feats, [args.num_hidden]*args.num_layers, n_classes, th.relu)
    model.to(device)
        # print(test(model, g))

    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)
    print(test(model, data))
    for epoch in range(args.num_epochs):
        model.train()
        tot_loss = 0
        tot_nodes = 0
        min_size = 1e12
        max_size = 0
        for bid, subg in enumerate(loader):
            min_size = min(min_size, subg.number_of_edges())
            max_size = max(max_size, subg.number_of_edges())
            optimizer.zero_grad()
            nodes = subg.ndata[dgl.NID]
            batch_train_mask = th.tensor(train_mask[nodes]).flatten()
            # print(batch_train_mask.nonzero().shape)
            labeled_nodes = subg.ndata[dgl.NID][th.nonzero(batch_train_mask)]
            batch_input, batch_labels = load_subtensor(g, nodes, labeled_nodes, device)
            batch_pred = model(subg, batch_input)
            batch_labled_pred = batch_pred[th.nonzero(batch_train_mask).flatten()]

            loss = F.nll_loss(batch_labled_pred, batch_labels.flatten())
            loss.backward()
            optimizer.step()
            # print(subg.number_of_nodes(), subg.number_of_edges())
            # print(loss.item())
            tot_loss += loss.item() * subg.number_of_nodes()
            tot_nodes += subg.number_of_nodes()
        print("#edge min {} max {}".format(min_size, max_size))
        print("epoch {} loss {}".format(epoch, tot_loss / tot_nodes))
        if ((epoch + 1) % 5 == 0):
            print(test(model, data))

@th.no_grad()
def test(model, data):
    train_mask, val_mask, in_feats, labels, n_classes, g = data
    model.eval()

    out = model(g, g.ndata['feature'].to(device))
    pred = out.argmax(dim=-1)
    correct = pred.eq(g.ndata['label'].to(device))

    accs = []
    for mask in [train_mask, val_mask]:
        accs.append(correct[mask].sum().item() / mask.sum().item())
    return accs

if __name__ == "__main__":
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=0,
        help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-epochs', type=int, default=50)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--num-workers', type=int, default=0)
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)

    args = argparser.parse_args()

    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    data = RedditDataset()
    # g = KarateClub()[0]
    g = dgl.graph(data.graph.all_edges(), readonly=True)
    
    train_mask = data.train_mask
    val_mask = data.val_mask
    features = th.Tensor(data.features)
    g.ndata['feature'] = features
    in_feats = features.shape[1]
    labels = th.LongTensor(data.labels)
    g.ndata['label'] = labels
    print(th.min(labels), th.max(labels))
    n_classes = data.num_labels

    data = train_mask, val_mask, in_feats, labels, n_classes, g
    run(args, device, data)
    # sampler = RandomWalkSampler(g, 1, 10)
    # model = SAINT(10, [10, 20], 2, F.relu)
    # feat = th.normal(0, 1, size=(g.number_of_nodes(), 10))

    # loss_fcn = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # batch_pred = model(g, feat)

    # print(model.forward(g, feat))