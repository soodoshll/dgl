import networkx as nx
import scipy.sparse as ssp
import dgl
import dgl.contrib as contrib
from dgl.frame import Frame, FrameRef, Column
from dgl.graph_index import create_graph_index
from dgl.utils import toindex
import backend as F
import dgl.function as fn
import pickle
import io
import unittest
from utils import parametrize_dtype
import multiprocessing as mp
import os

def create_test_graph(index_dtype):
    plays_spmat = ssp.coo_matrix(([1, 1, 1, 1], ([0, 1, 2, 1], [0, 0, 1, 1])))
    wishes_nx = nx.DiGraph()
    wishes_nx.add_nodes_from(['u0', 'u1', 'u2'], bipartite=0)
    wishes_nx.add_nodes_from(['g0', 'g1'], bipartite=1)
    wishes_nx.add_edge('u0', 'g1', id=0)
    wishes_nx.add_edge('u2', 'g0', id=1)

    follows_g = dgl.graph([(0, 1), (1, 2)], 'user', 'follows', index_dtype=index_dtype)
    plays_g = dgl.bipartite(plays_spmat, 'user', 'plays', 'game', index_dtype=index_dtype)
    wishes_g = dgl.bipartite(wishes_nx, 'user', 'wishes', 'game', index_dtype=index_dtype)
    develops_g = dgl.bipartite([(0, 0), (1, 1)], 'developer', 'develops', 'game', index_dtype=index_dtype)
    g = dgl.hetero_from_relations([follows_g, plays_g, wishes_g, develops_g])
    return g

def _assert_is_identical_hetero(g, g2):
    assert g.is_readonly == g2.is_readonly
    assert g.ntypes == g2.ntypes
    assert g.canonical_etypes == g2.canonical_etypes

    # check if two metagraphs are identical
    for edges, features in g.metagraph.edges(keys=True).items():
        assert g2.metagraph.edges(keys=True)[edges] == features

    # check if node ID spaces and feature spaces are equal
    for ntype in g.ntypes:
        assert g.number_of_nodes(ntype) == g2.number_of_nodes(ntype)

    # check if edge ID spaces and feature spaces are equal
    for etype in g.canonical_etypes:
        src, dst = g.all_edges(etype=etype, order='eid')
        src2, dst2 = g2.all_edges(etype=etype, order='eid')
        assert F.array_equal(src, src2)
        assert F.array_equal(dst, dst2)

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@parametrize_dtype
def test_single_process(index_dtype):
    hg = create_test_graph(index_dtype=index_dtype)
    hg_share = hg.shared_memory("hg")
    hg_rebuild = dgl.DGLHeteroGraph.create_heterograph_from_shared_memory('hg')
    hg_save_again = hg_rebuild.shared_memory("hg")
    _assert_is_identical_hetero(hg, hg_share)
    _assert_is_identical_hetero(hg, hg_rebuild)
    _assert_is_identical_hetero(hg, hg_save_again)

def sub_proc(hg_origin, name):
    hg_rebuild = dgl.DGLHeteroGraph.create_heterograph_from_shared_memory(name)
    hg_save_again = hg_rebuild.shared_memory(name)
    _assert_is_identical_hetero(hg_origin, hg_rebuild)
    _assert_is_identical_hetero(hg_origin, hg_save_again)

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@parametrize_dtype
def test_multi_process(index_dtype):
    hg = create_test_graph(index_dtype=index_dtype)
    hg_share = hg.shared_memory("hg1")
    p = mp.Process(target=sub_proc, args=(hg, "hg1"))
    p.start()
    p.join()

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@unittest.skipIf(F._default_context_str == 'cpu', reason="Need gpu for this test")
def test_copy_from_gpu():
    hg = create_test_graph(index_dtype="int32")
    hg_gpu = hg.copy_to(dgl.utils.to_dgl_context(F.cuda()))
    hg_share = hg_gpu.shared_memory("hg_gpu")
    p = mp.Process(target=sub_proc, args=(hg, "hg_gpu"))
    p.start()
    p.join()

if __name__ == "__main__":
    test_single_process("int64")
    # test_multi_process("int32")
    # test_copy_from_gpu()