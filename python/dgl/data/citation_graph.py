"""Cora, citeseer, pubmed dataset.

(lingfan): following dataset loading and preprocessing code from tkipf/gcn
https://github.com/tkipf/gcn/blob/master/gcn/utils.py
"""
from __future__ import absolute_import

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import os, sys

from .utils import download, extract_archive, get_download_dir, _get_dgl_url
from ..utils import retry_method_with_fix
from .. import convert
from .. import batch
from .. import backend as F

_urls = {
    'cora_v2' : 'dataset/cora_v2.zip',
    'citeseer' : 'dataset/citeseer.zip',
    'pubmed' : 'dataset/pubmed.zip',
    'cora_binary' : 'dataset/cora_binary.zip',
}

def _pickle_load(pkl_file):
    if sys.version_info > (3, 0):
        return pkl.load(pkl_file, encoding='latin1')
    else:
        return pkl.load(pkl_file)

class CitationGraphDataset(object):
    r"""The citation graph dataset, including cora, citeseer and pubmeb.
    Nodes mean authors and edges mean citation relationships.

    Parameters
    -----------
    name: str
      name can be 'cora', 'citeseer' or 'pubmed'.
    """
    def __init__(self, name):
        assert name.lower() in ['cora', 'citeseer', 'pubmed']

        # Previously we use the pre-processing in pygcn (https://github.com/tkipf/pygcn)
        # for Cora, which is slightly different from the one used in the GCN paper
        if name.lower() == 'cora':
            name = 'cora_v2'

        self.name = name
        self.dir = get_download_dir()
        self.zip_file_path='{}/{}.zip'.format(self.dir, name)
        self._load()

    def _download_and_extract(self):
        download(_get_dgl_url(_urls[self.name]), path=self.zip_file_path)
        extract_archive(self.zip_file_path, '{}/{}'.format(self.dir, self.name))

    @retry_method_with_fix(_download_and_extract)
    def _load(self):
        """Loads input data from gcn/data directory

        ind.name.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
        ind.name.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
        ind.name.allx => the feature vectors of both labeled and unlabeled training instances
            (a superset of ind.name.x) as scipy.sparse.csr.csr_matrix object;
        ind.name.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
        ind.name.ty => the one-hot labels of the test instances as numpy.ndarray object;
        ind.name.ally => the labels for instances in ind.name.allx as numpy.ndarray object;
        ind.name.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
            object;
        ind.name.test.index => the indices of test instances in graph, for the inductive setting as list object.

        All objects above must be saved using python pickle module.

        :param name: Dataset name
        :return: All data input files loaded (as well the training/test data).
        """
        root = '{}/{}'.format(self.dir, self.name)
        objnames = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(objnames)):
            with open("{}/ind.{}.{}".format(root, self.name, objnames[i]), 'rb') as f:
                objects.append(_pickle_load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = _parse_index_file("{}/ind.{}.test.index".format(root, self.name))
        test_idx_range = np.sort(test_idx_reorder)

        if self.name == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        graph = nx.DiGraph(nx.from_dict_of_lists(graph))

        onehot_labels = np.vstack((ally, ty))
        onehot_labels[test_idx_reorder, :] = onehot_labels[test_idx_range, :]
        labels = np.argmax(onehot_labels, 1)

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)

        train_mask = _sample_mask(idx_train, labels.shape[0])
        val_mask = _sample_mask(idx_val, labels.shape[0])
        test_mask = _sample_mask(idx_test, labels.shape[0])

        self.graph = graph
        self.features = _preprocess_features(features)
        self.labels = labels
        self.onehot_labels = onehot_labels
        self.num_labels = onehot_labels.shape[1]
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

        print('Finished data loading and preprocessing.')
        print('  NumNodes: {}'.format(self.graph.number_of_nodes()))
        print('  NumEdges: {}'.format(self.graph.number_of_edges()))
        print('  NumFeats: {}'.format(self.features.shape[1]))
        print('  NumClasses: {}'.format(self.num_labels))
        print('  NumTrainingSamples: {}'.format(len(np.nonzero(self.train_mask)[0])))
        print('  NumValidationSamples: {}'.format(len(np.nonzero(self.val_mask)[0])))
        print('  NumTestSamples: {}'.format(len(np.nonzero(self.test_mask)[0])))

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        g = convert.graph(self.graph)
        g.ndata['train_mask'] = F.tensor(self.train_mask, F.bool)
        g.ndata['val_mask'] = F.tensor(self.val_mask, F.bool)
        g.ndata['test_mask'] = F.tensor(self.test_mask, F.bool)
        g.ndata['label'] = F.tensor(self.labels, F.int64)
        g.ndata['feat'] = F.tensor(self.features, F.float32)
        return g

    def __len__(self):
        return 1

def _preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.asarray(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return np.asarray(features.todense())

def _parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def _sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask

def load_cora():
    data = CitationGraphDataset('cora')
    return data

def load_citeseer():
    data = CitationGraphDataset('citeseer')
    return data

def load_pubmed():
    data = CitationGraphDataset('pubmed')
    return data

class GCNSyntheticDataset(object):
    def __init__(self,
                 graph_generator,
                 num_feats=500,
                 num_classes=10,
                 train_ratio=1.,
                 val_ratio=0.,
                 test_ratio=0.,
                 seed=None):
        rng = np.random.RandomState(seed)
        # generate graph
        self.graph = graph_generator(seed)
        num_nodes = self.graph.number_of_nodes()

        # generate features
        #self.features = rng.randn(num_nodes, num_feats).astype(np.float32)
        self.features = np.zeros((num_nodes, num_feats), dtype=np.float32)

        # generate labels
        self.labels = rng.randint(num_classes, size=num_nodes)
        onehot_labels = np.zeros((num_nodes, num_classes), dtype=np.float32)
        onehot_labels[np.arange(num_nodes), self.labels] = 1.
        self.onehot_labels = onehot_labels
        self.num_labels = num_classes

        # generate masks
        ntrain = int(num_nodes * train_ratio)
        nval = int(num_nodes * val_ratio)
        ntest = int(num_nodes * test_ratio)
        mask_array = np.zeros((num_nodes,), dtype=np.int32)
        mask_array[0:ntrain] = 1
        mask_array[ntrain:ntrain+nval] = 2
        mask_array[ntrain+nval:ntrain+nval+ntest] = 3
        rng.shuffle(mask_array)
        self.train_mask = (mask_array == 1).astype(np.int32)
        self.val_mask = (mask_array == 2).astype(np.int32)
        self.test_mask = (mask_array == 3).astype(np.int32)

        print('Finished synthetic dataset generation.')
        print('  NumNodes: {}'.format(self.graph.number_of_nodes()))
        print('  NumEdges: {}'.format(self.graph.number_of_edges()))
        print('  NumFeats: {}'.format(self.features.shape[1]))
        print('  NumClasses: {}'.format(self.num_labels))
        print('  NumTrainingSamples: {}'.format(len(np.nonzero(self.train_mask)[0])))
        print('  NumValidationSamples: {}'.format(len(np.nonzero(self.val_mask)[0])))
        print('  NumTestSamples: {}'.format(len(np.nonzero(self.test_mask)[0])))

    def __getitem__(self, idx):
        return self

    def __len__(self):
        return 1

def get_gnp_generator(args):
    n = args.syn_gnp_n
    p = (2 * np.log(n) / n) if args.syn_gnp_p == 0. else args.syn_gnp_p
    def _gen(seed):
        return nx.fast_gnp_random_graph(n, p, seed, True)
    return _gen

class ScipyGraph(object):
    """A simple graph object that uses scipy matrix."""
    def __init__(self, mat):
        self._mat = mat

    def get_graph(self):
        return self._mat

    def number_of_nodes(self):
        return self._mat.shape[0]

    def number_of_edges(self):
        return self._mat.getnnz()

def get_scipy_generator(args):
    n = args.syn_gnp_n
    p = (2 * np.log(n) / n) if args.syn_gnp_p == 0. else args.syn_gnp_p
    def _gen(seed):
        return ScipyGraph(sp.random(n, n, p, format='coo'))
    return _gen

def load_synthetic(args):
    ty = args.syn_type
    if ty == 'gnp':
        gen = get_gnp_generator(args)
    elif ty == 'scipy':
        gen = get_scipy_generator(args)
    else:
        raise ValueError('Unknown graph generator type: {}'.format(ty))
    return GCNSyntheticDataset(
            gen,
            args.syn_nfeats,
            args.syn_nclasses,
            args.syn_train_ratio,
            args.syn_val_ratio,
            args.syn_test_ratio,
            args.syn_seed)

def register_args(parser):
    # Args for synthetic graphs.
    parser.add_argument('--syn-type', type=str, default='gnp',
            help='Type of the synthetic graph generator')
    parser.add_argument('--syn-nfeats', type=int, default=500,
            help='Number of node features')
    parser.add_argument('--syn-nclasses', type=int, default=10,
            help='Number of output classes')
    parser.add_argument('--syn-train-ratio', type=float, default=.1,
            help='Ratio of training nodes')
    parser.add_argument('--syn-val-ratio', type=float, default=.2,
            help='Ratio of validation nodes')
    parser.add_argument('--syn-test-ratio', type=float, default=.5,
            help='Ratio of testing nodes')
    # Args for GNP generator
    parser.add_argument('--syn-gnp-n', type=int, default=1000,
            help='n in gnp random graph')
    parser.add_argument('--syn-gnp-p', type=float, default=0.0,
            help='p in gnp random graph')
    parser.add_argument('--syn-seed', type=int, default=42,
            help='random seed')

class CoraBinary(object):
    """A mini-dataset for binary classification task using Cora.

    After loaded, it has following members:

    graphs : list of :class:`~dgl.DGLGraph`
    pmpds : list of :class:`scipy.sparse.coo_matrix`
    labels : list of :class:`numpy.ndarray`
    """
    def __init__(self):
        self.dir = get_download_dir()
        self.name = 'cora_binary'
        self.zip_file_path='{}/{}.zip'.format(self.dir, self.name)
        self._load()

    def _download_and_extract(self):
        download(_get_dgl_url(_urls[self.name]), path=self.zip_file_path)
        extract_archive(self.zip_file_path, '{}/{}'.format(self.dir, self.name))

    @retry_method_with_fix(_download_and_extract)
    def _load(self):
        root = '{}/{}'.format(self.dir, self.name)
        # load graphs
        self.graphs = []
        with open("{}/graphs.txt".format(root), 'r') as f:
            elist = []
            for line in f.readlines():
                if line.startswith('graph'):
                    if len(elist) != 0:
                        self.graphs.append(convert.graph(elist))
                    elist = []
                else:
                    u, v = line.strip().split(' ')
                    elist.append((int(u), int(v)))
            if len(elist) != 0:
                self.graphs.append(convert.graph(elist))
        with open("{}/pmpds.pkl".format(root), 'rb') as f:
            self.pmpds = _pickle_load(f)
        self.labels = []
        with open("{}/labels.txt".format(root), 'r') as f:
            cur = []
            for line in f.readlines():
                if line.startswith('graph'):
                    if len(cur) != 0:
                        self.labels.append(np.asarray(cur))
                    cur = []
                else:
                    cur.append(int(line.strip()))
            if len(cur) != 0:
                self.labels.append(np.asarray(cur))
        # sanity check
        assert len(self.graphs) == len(self.pmpds)
        assert len(self.graphs) == len(self.labels)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, i):
        return (self.graphs[i], self.pmpds[i], self.labels[i])

    @staticmethod
    def collate_fn(cur):
        graphs, pmpds, labels = zip(*cur)
        batched_graphs = batch.batch(graphs)
        batched_pmpds = sp.block_diag(pmpds)
        batched_labels = np.concatenate(labels, axis=0)
        return batched_graphs, batched_pmpds, batched_labels

def _normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.asarray(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def _encode_onehot(labels):
    classes = list(sorted(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.asarray(list(map(classes_dict.get, labels)),
                               dtype=np.int32)
    return labels_onehot
