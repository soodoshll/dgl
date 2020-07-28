"""Checking and logging utilities."""
# pylint: disable=invalid-name
from __future__ import absolute_import, division

from ..base import DGLError, dgl_warning
from .. import backend as F
from .internal import to_dgl_context

def prepare_tensor(g, data, name):
    """Convert the data to ID tensor and check its ID type and context.

    If the data is already in tensor type, raise error if its ID type
    and context does not match the graph's.
    Otherwise, convert it to tensor type of the graph's ID type and
    ctx and return.

    Parameters
    ----------
    g : DGLHeteroGraph
        Graph.
    data : int, iterable of int, tensor
        Data.
    name : str
        Name of the data.

    Returns
    -------
    Tensor
        Data in tensor object.
    """
    ret = None
    if F.is_tensor(data):
        if F.dtype(data) != g.idtype or F.context(data) != g.device:
            raise DGLError('Expect argument "{}" to have data type {} and device '
                           'context {}. But got {} and {}.'.format(
                               name, g.idtype, g.device, F.dtype(data), F.context(data)))
        ret = data
    else:
        ret = F.copy_to(F.tensor(data, g.idtype), g.device)

    if F.ndim(ret) != 1:
        raise DGLError('Expect a 1-D tensor for argument "{}". But got {}.'.format(
            name, ret))
    return ret

def prepare_tensor_dict(g, data, name):
    """Convert a dictionary of data to a dictionary of ID tensors.

    If calls ``prepare_tensor`` on each key-value pair.

    Parameters
    ----------
    g : DGLHeteroGraph
        Graph.
    data : dict[str, (int, iterable of int, tensor)]
        Data dict.
    name : str
        Name of the data.

    Returns
    -------
    dict[str, tensor]
    """
    return {key : prepare_tensor(g, val, '{}["{}"]'.format(name, key))
            for key, val in data.items()}

def check_all_same_idtype(glist, name):
    """Check all the graphs have the same idtype."""
    if len(glist) == 0:
        return
    idtype = glist[0].idtype
    for i, g in enumerate(glist):
        if g.idtype != idtype:
            raise DGLError('Expect {}[{}] to have {} type ID, but got {}.'.format(
                name, i, idtype, g.idtype))

def check_all_same_device(glist, name):
    """Check all the graphs have the same device."""
    if len(glist) == 0:
        return
    device = glist[0].device
    for i, g in enumerate(glist):
        if g.device != device:
            raise DGLError('Expect {}[{}] to be on device {}, but got {}.'.format(
                name, i, device, g.device))

def check_all_same_keys(dict_list, name):
    """Check all the dictionaries have the same set of keys."""
    if len(dict_list) == 0:
        return
    keys = dict_list[0].keys()
    for dct in dict_list:
        if keys != dct.keys():
            raise DGLError('Expect all {} to have the same set of keys, but got'
                           ' {} and {}.'.format(name, keys, dct.keys()))

def check_all_have_keys(dict_list, keys, name):
    """Check the dictionaries all have the given keys."""
    if len(dict_list) == 0:
        return
    keys = set(keys)
    for dct in dict_list:
        if not keys.issubset(dct.keys()):
            raise DGLError('Expect all {} to include keys {}, but got {}.'.format(
                name, keys, dct.keys()))

def check_all_same_schema(feat_dict_list, keys, name):
    """Check the features of the given keys all have the same schema.

    Suggest calling ``check_all_have_keys`` first.

    Parameters
    ----------
    feat_dict_list : list[dict[str, Tensor]]
        Feature dictionaries.
    keys : list[str]
        Keys
    name : str
        Name of this feature dict.
    """
    if len(feat_dict_list) == 0:
        return
    for fdict in feat_dict_list:
        for k in keys:
            t1 = feat_dict_list[0][k]
            t2 = fdict[k]
            if F.dtype(t1) != F.dtype(t2) or F.shape(t1)[1:] != F.shape(t2)[1:]:
                raise DGLError('Expect all features {}["{}"] to have the same data type'
                               ' and feature size, but got\n\t{} {}\nand\n\t{} {}.'.format(
                                   name, k, F.dtype(t1), F.shape(t1)[1:],
                                   F.dtype(t2), F.shape(t2)[1:]))

def to_int32_graph_if_on_gpu(g):
    """Convert to int32 graph if the input graph is on GPU."""
    # device_type 2 is an internal code for GPU
    if to_dgl_context(g.device).device_type == 2 and g.idtype == F.int64:
        dgl_warning('Automatically cast a GPU int64 graph to int32.\n'
                    ' To suppress the warning, call DGLGraph.int() first\n'
                    ' or specify the ``device`` argument when creating the graph.')
        return g.int()
    else:
        return g
