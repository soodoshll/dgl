"""Training GCN model on citation graphs."""
import argparse, time
import numpy as np
import networkx as nx
import mxnet as mx
from mxnet import gluon

import dgl
from dgl.data import register_data_args, load_data

from gcn import GCN
#from gcn_mp import GCN
#from gcn_spmv import GCN

def evaluate(model, features, labels, mask):
    pred = model(features).argmax(axis=1)
    accuracy = ((pred == labels) * mask).sum() / mask.sum().asscalar()
    return accuracy.asscalar()

def main(args):
    # load and preprocess dataset
    data = load_data(args)
    features = mx.nd.array(data.features)
    labels = mx.nd.array(data.labels)
    train_mask = mx.nd.array(data.train_mask)
    val_mask = mx.nd.array(data.val_mask)
    test_mask = mx.nd.array(data.test_mask)

    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              train_mask.sum().asscalar(),
              val_mask.sum().asscalar(),
              test_mask.sum().asscalar()))

    if args.gpu < 0:
        cuda = False
        ctx = mx.cpu(0)
    else:
        cuda = True
        ctx = mx.gpu(args.gpu)

    features = features.as_in_context(ctx)
    labels = labels.as_in_context(ctx)
    train_mask = train_mask.as_in_context(ctx)
    val_mask = val_mask.as_in_context(ctx)
    test_mask = test_mask.as_in_context(ctx)

    # create GCN model
    g = data.graph
    if args.self_loop:
        g.remove_edges_from(nx.selfloop_edges(g))
        g.add_edges_from(zip(g.nodes(), g.nodes()))
    g = dgl.graph(g).to(ctx)
    # normalization
    degs = g.in_degrees().astype('float32')
    norm = mx.nd.power(degs, -0.5)
    if cuda:
        norm = norm.as_in_context(ctx)
    g.ndata['norm'] = mx.nd.expand_dims(norm, 1)

    model = GCN(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                mx.nd.relu,
                args.dropout)
    model.initialize(ctx=ctx)
    n_train_samples = train_mask.sum().asscalar()
    loss_fcn = gluon.loss.SoftmaxCELoss()

    # use optimizer
    print(model.collect_params())
    trainer = gluon.Trainer(model.collect_params(), 'adam',
            {'learning_rate': args.lr, 'wd': args.weight_decay})

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        if epoch >= 3:
            t0 = time.time()
        # forward
        with mx.autograd.record():
            pred = model(features)
            loss = loss_fcn(pred, labels, mx.nd.expand_dims(train_mask, 1))
            loss = loss.sum() / n_train_samples

        loss.backward()
        trainer.step(batch_size=1)

        if epoch >= 3:
            loss.asscalar()
            dur.append(time.time() - t0)
            acc = evaluate(model, features, labels, val_mask)
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
                  "ETputs(KTEPS) {:.2f}". format(
                epoch, np.mean(dur), loss.asscalar(), acc, n_edges / np.mean(dur) / 1000))

    # test set accuracy
    acc = evaluate(model, features, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=3e-2,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()

    print(args)

    main(args)
