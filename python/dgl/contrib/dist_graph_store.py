import dgl
import numpy as np
import dgl.backend as F
from dgl.data.utils import load_graphs
from .._ffi.function import _init_api
from .. import utils

from ..network import _create_sender, _create_receiver
from ..network import _finalize_sender, _finalize_receiver
from ..network import _network_wait, _add_receiver_addr
from ..network import _receiver_wait, _sender_connect
from ..network import _send_ds_msg, _recv_ds_msg
from ..network import _clear_ds_msg
from ..network import KVMsgType, DistSampleMsg

import socket

class DistGraphStoreServer(object):
    def __init__(self, server_namebook, server_id, local_data, partition_book, num_clients, net_type='socket', queue_size=2*1024*1024*1024):
        self._server_namebook = server_namebook
        self._server_id = server_id
        self._local_g = load_graphs(local_data)[0][0]
        self._part_book = np.load(partition_book)
        self._global2local = self._part_book['global2local']

        self._ip = self._server_namebook[self._server_id][0]
        self._port = self._server_namebook[self._server_id][1]

        self._num_clients = num_clients
        self._sender = _create_sender(net_type, queue_size)
        self._receiver = _create_receiver(net_type, queue_size)
        self._client_namebook = {}


    def start(self):
        _receiver_wait(self._receiver, self._ip, self._port, self._num_clients)

        addr_list = []
        for i in range(self._num_clients):
            msg = _recv_ds_msg(self._receiver)
            assert msg.type == KVMsgType.IP_ID
            addr_list.append(msg.name)

        # Assign client ID to each client node
        addr_list.sort()
        for ID in range(len(addr_list)):
            self._client_namebook[ID] = addr_list[ID]

        _network_wait()

        for ID, addr in self._client_namebook.items():
            client_ip, client_port = addr.split(':')
            _add_receiver_addr(self._sender, client_ip, int(client_port), ID)

        _sender_connect(self._sender)

        if self._server_id == 0:
            for client_id in range(len(self._client_namebook)):
                msg = DistSampleMsg(
                    type=KVMsgType.IP_ID,
                    rank=self._server_id,
                    fanout=0,
                    name=str(client_id),
                    seed=None,
                    result=None,
                    c_ptr=None)
                _send_ds_msg(self._sender, msg, client_id)

        print('Distributed Sampling service %d start successfully! Listen for request ...' % self._server_id)

        # Service loop
        _CAPI_RemoteSamplingServerLoop(self._receiver)


    def neighbor_sample(self, seed, fanout):
        # This part should be done in C++
        pass

class DistGraphStore(object):
    def __init__(self, server_namebook, partition_book, queue_size=2*1024*1024*1024, net_type='socket'):
        self._server_namebook = server_namebook
        self._part_book = np.load(partition_book)
        self._num_parts = int(self._part_book['num_parts'])

        # It looks redundant.
        self._part_id = F.zerocopy_to_dgl_ndarray((F.tensor(self._part_book['part_id'])))
        self._client_id = -1
        self._num_servers = len(server_namebook)

        self._sender = _create_sender(net_type, queue_size)
        self._receiver = _create_receiver(net_type, queue_size)


    def neighbor_sample(self, seed, fanout):
        # This part should be done in C++
        _CAPI_RemoteSamplingReqeust(self._sender,
                                    self._client_id,
                                    self._num_parts,
                                    self._part_id, 
                                    F.zerocopy_to_dgl_ndarray(seed), 
                                    fanout)
    
    def connect(self):
        for ID, addr in self._server_namebook.items():
            server_ip = addr[0]
            server_port = addr[1]
            _add_receiver_addr(self._sender, server_ip, server_port, ID)
        _sender_connect(self._sender)

        self._addr = self._get_local_usable_addr()
        client_ip, client_port = self._addr.split(':')

        msg = DistSampleMsg(
            type=KVMsgType.IP_ID,
            rank=0, # a tmp client ID
            fanout=0,
            name=self._addr,
            seed=None,
            result=None,
            c_ptr=None)
        
        for server_id in range(self._num_servers):
            _send_ds_msg(self._sender, msg, server_id)

        _receiver_wait(self._receiver, client_ip, int(client_port), self._num_servers)

        msg = _recv_ds_msg(self._receiver)
        assert msg.rank == 0
        self._client_id = int(msg.name)
        print("Distributed Sampling Client %d connect to the server successfully!" % self._client_id)

    def shut_down(self):
        """Shut down all KVServer nodes.

        We usually invoke this API by just one client (e.g., client_0).
        """
        for server_id in range(self._num_servers):
            msg = DistSampleMsg(
                type=KVMsgType.FINAL,
                rank=self._client_id,
                fanout=0,
                name=None,
                seed=None,
                result=None,
                c_ptr=None)
            _send_ds_msg(self._sender, msg, server_id)

    def _get_local_usable_addr(self):
        """Get local available IP and port

        Return
        ------
        str
            IP address, e.g., '192.168.8.12:50051'
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # doesn't even have to be reachable
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except:
            IP = '127.0.0.1'
        finally:
            s.close()
        
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("",0))
        s.listen(1)
        port = s.getsockname()[1]
        s.close()

        return IP + ':' + str(port)

_init_api('dgl.network', __name__)
