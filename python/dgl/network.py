"""DGL Distributed Training Infrastructure."""
from __future__ import absolute_import

import time
from enum import Enum
from collections import namedtuple

import dgl.backend as F
from ._ffi.function import _init_api
from .nodeflow import NodeFlow
from . import utils

_init_api("dgl.network")


################################ Common Network Components ##################################

_WAIT_TIME_SEC = 3  # 3 seconds


def _network_wait():
    """Sleep for a few seconds
    """
    time.sleep(_WAIT_TIME_SEC)


def _create_sender(net_type, msg_queue_size=2*1024*1024*1024):
    """Create a Sender communicator via C api

    Parameters
    ----------
    net_type : str
        'socket' or 'mpi'
    msg_queue_size : int
        message queue size (2GB by default)
    """
    assert net_type in ('socket', 'mpi'), 'Unknown network type.'
    return _CAPI_DGLSenderCreate(net_type, msg_queue_size)


def _create_receiver(net_type, msg_queue_size=2*1024*1024*1024):
    """Create a Receiver communicator via C api

    Parameters
    ----------
    net_type : str
        'socket' or 'mpi'
    msg_queue_size : int
        message queue size (2GB by default)
    """
    assert net_type in ('socket', 'mpi'), 'Unknown network type.'
    return _CAPI_DGLReceiverCreate(net_type, msg_queue_size)


def _finalize_sender(sender):
    """Finalize Sender communicator

    Parameters
    ----------
    sender : ctypes.c_void_p
        C Sender handle
    """
    _CAPI_DGLFinalizeSender(sender)


def _finalize_receiver(receiver):
    """Finalize Receiver Communicator
    """
    _CAPI_DGLFinalizeReceiver(receiver)


def _add_receiver_addr(sender, ip_addr, port, recv_id):
    """Add Receiver IP address to namebook

    Parameters
    ----------
    sender : ctypes.c_void_p
        C Sender handle
    ip_addr : str
        IP address of Receiver
    port : int
        listen of Receiver
    recv_id : int
        Receiver ID
    """
    assert recv_id >= 0, 'recv_id cannot be a negative number.'
    _CAPI_DGLSenderAddReceiver(sender, ip_addr, int(port), int(recv_id))


def _sender_connect(sender):
    """Connect to all the Receiver

    Parameters
    ----------
    sender : ctypes.c_void_p
        C Sender handle
    """
    _CAPI_DGLSenderConnect(sender)


def _receiver_wait(receiver, ip_addr, port, num_sender):
    """Wait all Sender to connect.

    Parameters
    ----------
    receiver : ctypes.c_void_p
        C Receiver handle
    ip_addr : str
        IP address of Receiver
    port : int
        port of Receiver
    num_sender : int
        total number of Sender
    """
    assert num_sender >= 0, 'num_sender cannot be a negative number.'
    _CAPI_DGLReceiverWait(receiver, ip_addr, int(port), int(num_sender))


################################ Distributed Sampler Components ################################


def _send_nodeflow(sender, nodeflow, recv_id):
    """Send sampled subgraph (Nodeflow) to remote Receiver.

    Parameters
    ----------
    sender : ctypes.c_void_p
        C Sender handle
    nodeflow : NodeFlow
        NodeFlow object
    recv_id : int
        Receiver ID
    """
    assert recv_id >= 0, 'recv_id cannot be a negative number.'
    gidx = nodeflow._graph
    node_mapping = nodeflow._node_mapping.todgltensor()
    edge_mapping = nodeflow._edge_mapping.todgltensor()
    layers_offsets = utils.toindex(nodeflow._layer_offsets).todgltensor()
    flows_offsets = utils.toindex(nodeflow._block_offsets).todgltensor()
    _CAPI_SenderSendNodeFlow(sender,
                             int(recv_id),
                             gidx,
                             node_mapping,
                             edge_mapping,
                             layers_offsets,
                             flows_offsets)

def _send_sampler_end_signal(sender, recv_id):
    """Send an epoch-end signal to remote Receiver.

    Parameters
    ----------
    sender : ctypes.c_void_p
        C sender handle
    recv_id : int
        Receiver ID
    """
    assert recv_id >= 0, 'recv_id cannot be a negative number.'
    _CAPI_SenderSendSamplerEndSignal(sender, int(recv_id))

def _recv_nodeflow(receiver, graph):
    """Receive sampled subgraph (NodeFlow) from remote sampler.

    Parameters
    ----------
    receiver : ctypes.c_void_p
        C Receiver handle
    graph : DGLGraph
        The parent graph

    Returns
    -------
    NodeFlow or an end-signal
    """
    res = _CAPI_ReceiverRecvNodeFlow(receiver)
    if isinstance(res, int):
        return res
    else:
        return NodeFlow(graph, res)


################################ Distributed KVStore Components ################################


class KVMsgType(Enum):
    """Type of kvstore message
    """
    FINAL = 1
    INIT = 2
    PUSH = 3
    PULL = 4
    PULL_BACK = 5
    BARRIER = 6
    IP_ID = 7

    SAMPLE_REQUEST = 8
    SAMPLE_RESPONSE = 9


KVStoreMsg = namedtuple("KVStoreMsg", "type rank name id data c_ptr")
"""Message of DGL kvstore

Data Field
----------
type : KVMsgType
    Type of DGL kvstore message
rank : int
    sender's ID
name : str
    data name
id : tensor (mx.ndarray or torch.tensor)
    data vector storing the global IDs
data : tensor (mx.ndarray or torch.tensor)
    data matrix with the same row size of id
c_ptr : void*
    c pointer of message
"""

def _send_kv_msg(sender, msg, recv_id):
    """Send kvstore message.

    Parameters
    ----------
    sender : ctypes.c_void_p
        C sender handle
    msg : KVStoreMsg
        kvstore message
    recv_id : int
        receiver's ID
    """
    if msg.type == KVMsgType.PULL:
        tensor_id = F.zerocopy_to_dgl_ndarray(msg.id)
        _CAPI_SenderSendKVMsg(
            sender,
            int(recv_id),
            msg.type.value,
            msg.rank,
            msg.name,
            tensor_id)
    elif msg.type == KVMsgType.IP_ID:
        _CAPI_SenderSendKVMsg(
            sender,
            int(recv_id),
            msg.type.value,
            msg.rank,
            msg.name)
    elif msg.type in (KVMsgType.FINAL, KVMsgType.BARRIER):
        _CAPI_SenderSendKVMsg(
            sender,
            int(recv_id),
            msg.type.value,
            msg.rank)
    else:
        tensor_id = F.zerocopy_to_dgl_ndarray(msg.id)
        data = F.zerocopy_to_dgl_ndarray(msg.data)
        _CAPI_SenderSendKVMsg(
            sender,
            int(recv_id),
            msg.type.value,
            msg.rank,
            msg.name,
            tensor_id,
            data)


def _recv_kv_msg(receiver):
    """Receive kvstore message.

    Parameters
    ----------
    receiver : ctypes.c_void_p
        C Receiver handle
    Return
    ------
    KVStoreMsg
        kvstore message
    """
    msg_ptr = CAPI_ReceiverRecvKVMsg(receiver)
    msg_type = KVMsgType(_CAPI_ReceiverGetKVMsgType(msg_ptr))
    rank = _CAPI_ReceiverGetKVMsgRank(msg_ptr)
    if msg_type == KVMsgType.PULL:
        name = _CAPI_ReceiverGetKVMsgName(msg_ptr)
        tensor_id = F.zerocopy_from_dgl_ndarray(_CAPI_ReceiverGetKVMsgID(msg_ptr))
        msg = KVStoreMsg(
            type=msg_type,
            rank=rank,
            name=name,
            id=tensor_id,
            data=None,
            c_ptr=msg_ptr)
        return msg
    elif msg_type == KVMsgType.IP_ID:
        name = _CAPI_ReceiverGetKVMsgName(msg_ptr)
        msg = KVStoreMsg(
            type=msg_type,
            rank=rank,
            name=name,
            id=None,
            data=None,
            c_ptr=msg_ptr)
        return msg
    elif msg_type in (KVMsgType.FINAL, KVMsgType.BARRIER):
        msg = KVStoreMsg(
            type=msg_type,
            rank=rank,
            name=None,
            id=None,
            data=None,
            c_ptr=msg_ptr)
        return msg
    else:
        name = _CAPI_ReceiverGetKVMsgName(msg_ptr)
        tensor_id = F.zerocopy_from_dgl_ndarray(_CAPI_ReceiverGetKVMsgID(msg_ptr))
        data = F.zerocopy_from_dgl_ndarray(_CAPI_ReceiverGetKVMsgData(msg_ptr))
        msg = KVStoreMsg(
            type=msg_type,
            rank=rank,
            name=name,
            id=tensor_id,
            data=data,
            c_ptr=msg_ptr)
        return msg

    raise RuntimeError('Unknown message type: %d' % msg_type.value)


def _clear_kv_msg(msg):
    """Clear data of kvstore message
    """
    F.sync()
    if msg.c_ptr is not None:
        _CAPI_DeleteKVMsg(msg.c_ptr)


################################ Distributed Sampling ################################

DistSampleMsg = namedtuple("DistSampleMsg", "type rank fanout name seed result c_ptr")
def _send_ds_msg(sender, msg, recv_id):
    """Send kvstore message.

    Parameters
    ----------
    sender : ctypes.c_void_p
        C sender handle
    msg : KVStoreMsg
        kvstore message
    recv_id : int
        receiver's ID
    """
    if msg.type == KVMsgType.SAMPLE_REQUEST:
        tensor_seed = F.zerocopy_to_dgl_ndarray(msg.seed)
        _CAPI_SenderSendDSMsg(
            sender,
            int(recv_id),
            msg.type.value,
            msg.rank,
            msg.fanout,
            msg.name,
            tensor_seed)
    elif msg.type == KVMsgType.SAMPLE_RESPONSE:
        tensor_result = F.zerocopy_to_dgl_ndarray(msg.result)
        _CAPI_SenderSendDSMsg(
            sender,
            int(recv_id),
            msg.type.value,
            msg.rank,
            0,
            msg.name,
            tensor_result)
    elif msg.type == KVMsgType.IP_ID:
        _CAPI_SenderSendDSMsg(
            sender,
            int(recv_id),
            msg.type.value,
            msg.rank,
            0,
            msg.name)
    elif msg.type in (KVMsgType.FINAL, KVMsgType.BARRIER):
        _CAPI_SenderSendDSMsg(
            sender,
            int(recv_id),
            msg.type.value,
            msg.rank,
            0)
    else:
        _CAPI_SenderSendDSMsg(
            sender,
            int(recv_id),
            msg.type.value,
            msg.rank,
            msg.name)


def _recv_ds_msg(receiver):
    """Receive kvstore message.

    Parameters
    ----------
    receiver : ctypes.c_void_p
        C Receiver handle
    Return
    ------
    KVStoreMsg
        kvstore message
    """
    msg_ptr = CAPI_ReceiverRecvDSMsg(receiver)
    msg_type = KVMsgType(_CAPI_ReceiverGetDSMsgType(msg_ptr))
    rank = _CAPI_ReceiverGetDSMsgRank(msg_ptr)
    fanout = _CAPI_ReceiverGetDSMsgFanout(msg_ptr)
    if msg_type == KVMsgType.SAMPLE_REQUEST:
        name = _CAPI_ReceiverGetDSMsgName(msg_ptr)
        tensor_seed = F.zerocopy_from_dgl_ndarray(_CAPI_ReceiverGetDSMsgSeed(msg_ptr))
        msg = DistSampleMsg(
            type=msg_type,
            rank=rank,
            fanout=fanout,
            name=name,
            seed=tensor_seed,
            result=None,
            c_ptr=msg_ptr)
        return msg
    elif msg_type == KVMsgType.SAMPLE_RESPONSE:
        name = _CAPI_ReceiverGetDSMsgName(msg_ptr)
        tensor_result = F.zerocopy_from_dgl_ndarray(_CAPI_ReceiverGetDSMsgResult(msg_ptr))
        msg = DistSampleMsg(
            type=msg_type,
            rank=rank,
            fanout=0,
            name=name,
            seed=None,
            result=tensor_result,
            c_ptr=msg_ptr)
        return msg
    elif msg_type == KVMsgType.IP_ID:
        name = _CAPI_ReceiverGetDSMsgName(msg_ptr)
        msg = DistSampleMsg(
            type=msg_type,
            rank=rank,
            fanout=0,
            name=name,
            seed=None,
            result=None,
            c_ptr=msg_ptr)
        return msg
    elif msg_type in (KVMsgType.FINAL, KVMsgType.BARRIER):
        msg = DistSampleMsg(
            type=msg_type,
            rank=rank,
            fanout=0,
            name=None,
            seed=None,
            result=None,
            c_ptr=msg_ptr)
        return msg
    else:
        name = _CAPI_ReceiverGetKVMsgName(msg_ptr)
        msg = DistSampleMsg(
            type=msg_type,
            rank=rank,
            fanout=0,
            name=name,
            seed=None,
            result=None,
            c_ptr=msg_ptr)
        return msg

    raise RuntimeError('Unknown message type: %d' % msg_type.value)


def _clear_ds_msg(msg):
    """Clear data of kvstore message
    """
    F.sync()
    if msg.c_ptr is not None:
        _CAPI_DeleteDSMsg(msg.c_ptr)