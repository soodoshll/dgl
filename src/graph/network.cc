/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/network.cc
 * \brief DGL networking related APIs
 */
#include "./network.h"

#include <dgl/runtime/container.h>
#include <dgl/runtime/ndarray.h>
#include <dgl/packed_func_ext.h>
#include <dgl/immutable_graph.h>
#include <dgl/nodeflow.h>
#include <dgl/sampling/neighbor.h>
#include <dmlc/omp.h>

#include <unordered_map>
#include <thread>

#include "./network/communicator.h"
#include "./network/socket_communicator.h"
#include "./network/msg_queue.h"
#include "./network/common.h"

#include "./heterograph.h"

using dgl::network::StringPrintf;
using namespace dgl::runtime;

namespace dgl {
namespace network {

static void NaiveDeleter(DLManagedTensor* managed_tensor) {
  delete [] managed_tensor->dl_tensor.shape;
  delete [] managed_tensor->dl_tensor.strides;
  delete [] managed_tensor->dl_tensor.data;
  delete managed_tensor;
}

NDArray CreateNDArrayFromRaw(std::vector<int64_t> shape,
                             DLDataType dtype,
                             DLContext ctx,
                             void* raw) {
  DLTensor tensor;
  tensor.ctx = ctx;
  tensor.ndim = static_cast<int>(shape.size());
  tensor.dtype = dtype;
  tensor.shape = new int64_t[tensor.ndim];
  for (int i = 0; i < tensor.ndim; ++i) {
    tensor.shape[i] = shape[i];
  }
  tensor.strides = new int64_t[tensor.ndim];
  for (int i = 0; i < tensor.ndim; ++i) {
    tensor.strides[i] = 1;
  }
  for (int i = tensor.ndim - 2; i >= 0; --i) {
    tensor.strides[i] = tensor.shape[i+1] * tensor.strides[i+1];
  }
  tensor.data = raw;
  DLManagedTensor *managed_tensor = new DLManagedTensor();
  managed_tensor->dl_tensor = tensor;
  managed_tensor->deleter = NaiveDeleter;
  return NDArray::FromDLPack(managed_tensor);
}

void ArrayMeta::AddArray(const NDArray& array) {
  // We first write the ndim to the data_shape_
  data_shape_.push_back(static_cast<int64_t>(array->ndim));
  // Then we write the data shape
  for (int i = 0; i < array->ndim; ++i) {
    data_shape_.push_back(array->shape[i]);
  }
  ndarray_count_++;
}

char* ArrayMeta::Serialize(int64_t* size) {
  char* buffer = nullptr;
  int64_t buffer_size = 0;
  buffer_size += sizeof(msg_type_);
  if (ndarray_count_ != 0) {
    buffer_size += sizeof(ndarray_count_);
    buffer_size += sizeof(data_shape_.size());
    buffer_size += sizeof(int64_t) * data_shape_.size();
  }
  // In the future, we should have a better memory management as
  // allocating a large chunk of memory can be very expensive.
  buffer = new char[buffer_size];
  char* pointer = buffer;
  // Write msg_type_
  *(reinterpret_cast<int*>(pointer)) = msg_type_;
  pointer += sizeof(msg_type_);
  if (ndarray_count_ != 0) {
    // Write ndarray_count_
    *(reinterpret_cast<int*>(pointer)) = ndarray_count_;
    pointer += sizeof(ndarray_count_);
    // Write size of data_shape_
    *(reinterpret_cast<size_t*>(pointer)) = data_shape_.size();
    pointer += sizeof(data_shape_.size());
    // Write data of data_shape_
    memcpy(pointer,
        reinterpret_cast<char*>(data_shape_.data()),
        sizeof(int64_t) * data_shape_.size());
  }
  *size = buffer_size;
  return buffer;
}

void ArrayMeta::Save(dmlc::Stream *stream) const{
  stream->Write(msg_type_);
  stream->Write(ndarray_count_);
  stream->Write(data_shape_);
}

void ArrayMeta::Load(dmlc::Stream *stream){
  stream->Read(&msg_type_);
  stream->Read(&ndarray_count_);
  stream->Write(data_shape_);
}

void ArrayMeta::Deserialize(char* buffer, int64_t size) {
  int64_t data_size = 0;
  // Read mesg_type_
  msg_type_ = *(reinterpret_cast<int*>(buffer));
  buffer += sizeof(int);
  data_size += sizeof(int);
  if (data_size < size) {
    // Read ndarray_count_
    ndarray_count_ = *(reinterpret_cast<int*>(buffer));
    buffer += sizeof(int);
    data_size += sizeof(int);
    // Read size of data_shape_
    size_t count = *(reinterpret_cast<size_t*>(buffer));
    buffer += sizeof(size_t);
    data_size += sizeof(size_t);
    data_shape_.resize(count);
    // Read data of data_shape_
    memcpy(data_shape_.data(), buffer,
        count * sizeof(int64_t));
    data_size += count * sizeof(int64_t);
  }
  CHECK_EQ(data_size, size);
}

char* KVStoreMsg::Serialize(int64_t* size) {
  char* buffer = nullptr;
  int64_t buffer_size = 0;
  buffer_size += sizeof(this->msg_type);
  buffer_size += sizeof(this->rank);
  if (!this->name.empty()) {
    buffer_size += sizeof(this->name.size());
    buffer_size += this->name.size();
  }
  // In the future, we should have a better memory management as
  // allocating a large chunk of memory can be very expensive.
  buffer = new char[buffer_size];
  char* pointer = buffer;
  // write msg_type
  *(reinterpret_cast<int*>(pointer)) = this->msg_type;
  pointer += sizeof(this->msg_type);
  // write rank
  *(reinterpret_cast<int*>(pointer)) = this->rank;
  pointer += sizeof(this->rank);
  // write name
  if (!this->name.empty()) {
    *(reinterpret_cast<size_t*>(pointer)) = this->name.size();
    pointer += sizeof(size_t);
    memcpy(pointer, this->name.c_str(), this->name.size());
  }
  *size = buffer_size;
  return buffer;
}

void KVStoreMsg::Deserialize(char* buffer, int64_t size) {
  int64_t data_size = 0;
  // Read msg_type
  this->msg_type = *(reinterpret_cast<int*>(buffer));
  buffer += sizeof(int);
  data_size += sizeof(int);
  // Read rank
  this->rank = *(reinterpret_cast<int*>(buffer));
  buffer += sizeof(int);
  data_size += sizeof(int);
  if (data_size < size) {
    // Read name
    size_t name_size = *(reinterpret_cast<size_t*>(buffer));
    buffer += sizeof(name_size);
    data_size += sizeof(name_size);
    this->name.assign(buffer, name_size);
    data_size += name_size;
  }
  CHECK_EQ(data_size, size);
}

////////////////////////////////// Basic Networking Components ////////////////////////////////


DGL_REGISTER_GLOBAL("network._CAPI_DGLSenderCreate")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string type = args[0];
    int64_t msg_queue_size = args[1];
    network::Sender* sender = nullptr;
    if (type == "socket") {
      sender = new network::SocketSender(msg_queue_size);
    } else {
      LOG(FATAL) << "Unknown communicator type: " << type;
    }
    CommunicatorHandle chandle = static_cast<CommunicatorHandle>(sender);
    *rv = chandle;
  });

DGL_REGISTER_GLOBAL("network._CAPI_DGLReceiverCreate")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string type = args[0];
    int64_t msg_queue_size = args[1];
    network::Receiver* receiver = nullptr;
    if (type == "socket") {
      receiver = new network::SocketReceiver(msg_queue_size);
    } else {
      LOG(FATAL) << "Unknown communicator type: " << type;
    }
    CommunicatorHandle chandle = static_cast<CommunicatorHandle>(receiver);
    *rv = chandle;
  });

DGL_REGISTER_GLOBAL("network._CAPI_DGLFinalizeSender")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    network::Sender* sender = static_cast<network::Sender*>(chandle);
    sender->Finalize();
  });

DGL_REGISTER_GLOBAL("network._CAPI_DGLFinalizeReceiver")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    network::Receiver* receiver = static_cast<network::SocketReceiver*>(chandle);
    receiver->Finalize();
  });

DGL_REGISTER_GLOBAL("network._CAPI_DGLSenderAddReceiver")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    std::string ip = args[1];
    int port = args[2];
    int recv_id = args[3];
    network::Sender* sender = static_cast<network::Sender*>(chandle);
    std::string addr;
    if (sender->Type() == "socket") {
      addr = StringPrintf("socket://%s:%d", ip.c_str(), port);
    } else {
      LOG(FATAL) << "Unknown communicator type: " << sender->Type();
    }
    sender->AddReceiver(addr.c_str(), recv_id);
  });

DGL_REGISTER_GLOBAL("network._CAPI_DGLSenderConnect")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    network::Sender* sender = static_cast<network::Sender*>(chandle);
    if (sender->Connect() == false) {
      LOG(FATAL) << "Sender connection failed.";
    }
  });

DGL_REGISTER_GLOBAL("network._CAPI_DGLReceiverWait")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    std::string ip = args[1];
    int port = args[2];
    int num_sender = args[3];
    network::Receiver* receiver = static_cast<network::SocketReceiver*>(chandle);
    std::string addr;
    if (receiver->Type() == "socket") {
      addr = StringPrintf("socket://%s:%d", ip.c_str(), port);
    } else {
      LOG(FATAL) << "Unknown communicator type: " << receiver->Type();
    }
    if (receiver->Wait(addr.c_str(), num_sender) == false) {
      LOG(FATAL) << "Wait sender socket failed.";
    }
  });


////////////////////////// Distributed Sampler Components ////////////////////////////////


DGL_REGISTER_GLOBAL("network._CAPI_SenderSendNodeFlow")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    int recv_id = args[1];
    GraphRef g = args[2];
    NDArray node_mapping = args[3];
    NDArray edge_mapping = args[4];
    NDArray layer_offsets = args[5];
    NDArray flow_offsets = args[6];
    auto ptr = std::dynamic_pointer_cast<ImmutableGraph>(g.sptr());
    CHECK(ptr) << "only immutable graph is allowed in send/recv";
    auto csr = ptr->GetInCSR();
    // Create a message for the meta data of ndarray
    NDArray indptr = csr->indptr();
    NDArray indice = csr->indices();
    NDArray edge_ids = csr->edge_ids();
    ArrayMeta meta(kNodeFlowMsg);
    meta.AddArray(node_mapping);
    meta.AddArray(edge_mapping);
    meta.AddArray(layer_offsets);
    meta.AddArray(flow_offsets);
    meta.AddArray(indptr);
    meta.AddArray(indice);
    meta.AddArray(edge_ids);
    // send meta message
    int64_t size = 0;
    char* data = meta.Serialize(&size);
    network::Sender* sender = static_cast<network::Sender*>(chandle);
    Message send_msg;
    send_msg.data = data;
    send_msg.size = size;
    send_msg.deallocator = DefaultMessageDeleter;
    CHECK_EQ(sender->Send(send_msg, recv_id), ADD_SUCCESS);
    // send node_mapping
    Message node_mapping_msg;
    node_mapping_msg.data = static_cast<char*>(node_mapping->data);
    node_mapping_msg.size = node_mapping.GetSize();
    // capture the array in the closure
    node_mapping_msg.deallocator = [node_mapping](Message*) {};
    CHECK_EQ(sender->Send(node_mapping_msg, recv_id), ADD_SUCCESS);
    // send edege_mapping
    Message edge_mapping_msg;
    edge_mapping_msg.data = static_cast<char*>(edge_mapping->data);
    edge_mapping_msg.size = edge_mapping.GetSize();
    // capture the array in the closure
    edge_mapping_msg.deallocator = [edge_mapping](Message*) {};
    CHECK_EQ(sender->Send(edge_mapping_msg, recv_id), ADD_SUCCESS);
    // send layer_offsets
    Message layer_offsets_msg;
    layer_offsets_msg.data = static_cast<char*>(layer_offsets->data);
    layer_offsets_msg.size = layer_offsets.GetSize();
    // capture the array in the closure
    layer_offsets_msg.deallocator = [layer_offsets](Message*) {};
    CHECK_EQ(sender->Send(layer_offsets_msg, recv_id), ADD_SUCCESS);
    // send flow_offset
    Message flow_offsets_msg;
    flow_offsets_msg.data = static_cast<char*>(flow_offsets->data);
    flow_offsets_msg.size = flow_offsets.GetSize();
    // capture the array in the closure
    flow_offsets_msg.deallocator = [flow_offsets](Message*) {};
    CHECK_EQ(sender->Send(flow_offsets_msg, recv_id), ADD_SUCCESS);
    // send csr->indptr
    Message indptr_msg;
    indptr_msg.data = static_cast<char*>(indptr->data);
    indptr_msg.size = indptr.GetSize();
    // capture the array in the closure
    indptr_msg.deallocator = [indptr](Message*) {};
    CHECK_EQ(sender->Send(indptr_msg, recv_id), ADD_SUCCESS);
    // send csr->indices
    Message indices_msg;
    indices_msg.data = static_cast<char*>(indice->data);
    indices_msg.size = indice.GetSize();
    // capture the array in the closure
    indices_msg.deallocator = [indice](Message*) {};
    CHECK_EQ(sender->Send(indices_msg, recv_id), ADD_SUCCESS);
    // send csr->edge_ids
    Message edge_ids_msg;
    edge_ids_msg.data = static_cast<char*>(edge_ids->data);
    edge_ids_msg.size = edge_ids.GetSize();
    // capture the array in the closure
    edge_ids_msg.deallocator = [edge_ids](Message*) {};
    CHECK_EQ(sender->Send(edge_ids_msg, recv_id), ADD_SUCCESS);
  });

DGL_REGISTER_GLOBAL("network._CAPI_SenderSendSamplerEndSignal")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    int recv_id = args[1];
    ArrayMeta meta(kFinalMsg);
    int64_t size = 0;
    char* data = meta.Serialize(&size);
    network::Sender* sender = static_cast<network::Sender*>(chandle);
    Message send_msg = {data, size};
    send_msg.deallocator = DefaultMessageDeleter;
    CHECK_EQ(sender->Send(send_msg, recv_id), ADD_SUCCESS);
  });

DGL_REGISTER_GLOBAL("network._CAPI_ReceiverRecvNodeFlow")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    network::Receiver* receiver = static_cast<network::SocketReceiver*>(chandle);
    int send_id = 0;
    Message recv_msg;
    CHECK_EQ(receiver->Recv(&recv_msg, &send_id), REMOVE_SUCCESS);
    ArrayMeta meta(recv_msg.data, recv_msg.size);
    recv_msg.deallocator(&recv_msg);
    if (meta.msg_type() == kNodeFlowMsg) {
      CHECK_EQ(meta.ndarray_count() * 2, meta.data_shape_.size());
      NodeFlow nf = NodeFlow::Create();
      // node_mapping
      Message array_0;
      CHECK_EQ(receiver->RecvFrom(&array_0, send_id), REMOVE_SUCCESS);
      CHECK_EQ(meta.data_shape_[0], 1);
      nf->node_mapping = CreateNDArrayFromRaw(
        {meta.data_shape_[1]},
        DLDataType{kDLInt, 64, 1},
        DLContext{kDLCPU, 0},
        array_0.data);
      // edge_mapping
      Message array_1;
      CHECK_EQ(receiver->RecvFrom(&array_1, send_id), REMOVE_SUCCESS);
      CHECK_EQ(meta.data_shape_[2], 1);
      nf->edge_mapping = CreateNDArrayFromRaw(
        {meta.data_shape_[3]},
        DLDataType{kDLInt, 64, 1},
        DLContext{kDLCPU, 0},
        array_1.data);
      // layer_offset
      Message array_2;
      CHECK_EQ(receiver->RecvFrom(&array_2, send_id), REMOVE_SUCCESS);
      CHECK_EQ(meta.data_shape_[4], 1);
      nf->layer_offsets = CreateNDArrayFromRaw(
        {meta.data_shape_[5]},
        DLDataType{kDLInt, 64, 1},
        DLContext{kDLCPU, 0},
        array_2.data);
      // flow_offset
      Message array_3;
      CHECK_EQ(receiver->RecvFrom(&array_3, send_id), REMOVE_SUCCESS);
      CHECK_EQ(meta.data_shape_[6], 1);
      nf->flow_offsets = CreateNDArrayFromRaw(
        {meta.data_shape_[7]},
        DLDataType{kDLInt, 64, 1},
        DLContext{kDLCPU, 0},
        array_3.data);
      // CSR indptr
      Message array_4;
      CHECK_EQ(receiver->RecvFrom(&array_4, send_id), REMOVE_SUCCESS);
      CHECK_EQ(meta.data_shape_[8], 1);
      NDArray indptr = CreateNDArrayFromRaw(
        {meta.data_shape_[9]},
        DLDataType{kDLInt, 64, 1},
        DLContext{kDLCPU, 0},
        array_4.data);
      // CSR indice
      Message array_5;
      CHECK_EQ(receiver->RecvFrom(&array_5, send_id), REMOVE_SUCCESS);
      CHECK_EQ(meta.data_shape_[10], 1);
      NDArray indice = CreateNDArrayFromRaw(
        {meta.data_shape_[11]},
        DLDataType{kDLInt, 64, 1},
        DLContext{kDLCPU, 0},
        array_5.data);
      // CSR edge_ids
      Message array_6;
      CHECK_EQ(receiver->RecvFrom(&array_6, send_id), REMOVE_SUCCESS);
      CHECK_EQ(meta.data_shape_[12], 1);
      NDArray edge_ids = CreateNDArrayFromRaw(
        {meta.data_shape_[13]},
        DLDataType{kDLInt, 64, 1},
        DLContext{kDLCPU, 0},
        array_6.data);
      // Create CSR
      CSRPtr csr(new CSR(indptr, indice, edge_ids));
      nf->graph = GraphPtr(new ImmutableGraph(csr, nullptr));
      *rv = nf;
    } else if (meta.msg_type() == kFinalMsg) {
      *rv = meta.msg_type();
    } else {
      LOG(FATAL) << "Unknown message type: " << meta.msg_type();
    }
  });


////////////////////////// Distributed KVStore Components ////////////////////////////////


DGL_REGISTER_GLOBAL("network._CAPI_SenderSendKVMsg")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    int args_count = 0;
    CommunicatorHandle chandle = args[args_count++];
    int recv_id = args[args_count++];
    KVStoreMsg kv_msg;
    kv_msg.msg_type = args[args_count++];
    kv_msg.rank = args[args_count++];
    network::Sender* sender = static_cast<network::Sender*>(chandle);
    if (kv_msg.msg_type != kFinalMsg && kv_msg.msg_type != kBarrierMsg) {
      std::string name = args[args_count++];
      kv_msg.name = name;
      if (kv_msg.msg_type != kIPIDMsg) {
        kv_msg.id = args[args_count++];
      }
      if (kv_msg.msg_type != kPullMsg && kv_msg.msg_type != kIPIDMsg) {
        kv_msg.data = args[args_count++];
      }
    }
    int64_t kv_size = 0;
    char* kv_data = kv_msg.Serialize(&kv_size);
    // Send kv_data
    Message send_kv_msg;
    send_kv_msg.data = kv_data;
    send_kv_msg.size = kv_size;
    send_kv_msg.deallocator = DefaultMessageDeleter;
    CHECK_EQ(sender->Send(send_kv_msg, recv_id), ADD_SUCCESS);

    if (kv_msg.msg_type != kFinalMsg &&
        kv_msg.msg_type != kBarrierMsg &&
        kv_msg.msg_type != kIPIDMsg) {
      // Send ArrayMeta
      ArrayMeta meta(kv_msg.msg_type);
      meta.AddArray(kv_msg.id);
      if (kv_msg.msg_type != kPullMsg) {
        meta.AddArray(kv_msg.data);
      }
      int64_t meta_size = 0;
      char* meta_data = meta.Serialize(&meta_size);
      Message send_meta_msg;
      send_meta_msg.data = meta_data;
      send_meta_msg.size = meta_size;
      send_meta_msg.deallocator = DefaultMessageDeleter;
      CHECK_EQ(sender->Send(send_meta_msg, recv_id), ADD_SUCCESS);
      // Send ID NDArray
      Message send_id_msg;
      send_id_msg.data = static_cast<char*>(kv_msg.id->data);
      send_id_msg.size = kv_msg.id.GetSize();
      NDArray id = kv_msg.id;
      send_id_msg.deallocator = [id](Message*) {};
      CHECK_EQ(sender->Send(send_id_msg, recv_id), ADD_SUCCESS);
      // Send data NDArray
      if (kv_msg.msg_type != kPullMsg) {
        Message send_data_msg;
        send_data_msg.data = static_cast<char*>(kv_msg.data->data);
        send_data_msg.size = kv_msg.data.GetSize();
        NDArray data = kv_msg.data;
        send_data_msg.deallocator = [data](Message*) {};
        CHECK_EQ(sender->Send(send_data_msg, recv_id), ADD_SUCCESS);
      }
    }
  });

DGL_REGISTER_GLOBAL("network.CAPI_ReceiverRecvKVMsg")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    network::Receiver* receiver = static_cast<network::SocketReceiver*>(chandle);
    KVStoreMsg *kv_msg = new KVStoreMsg();
    // Recv kv_Msg
    Message recv_kv_msg;
    int send_id;
    CHECK_EQ(receiver->Recv(&recv_kv_msg, &send_id), REMOVE_SUCCESS);
    kv_msg->Deserialize(recv_kv_msg.data, recv_kv_msg.size);
    recv_kv_msg.deallocator(&recv_kv_msg);
    if (kv_msg->msg_type == kFinalMsg ||
        kv_msg->msg_type == kBarrierMsg ||
        kv_msg->msg_type == kIPIDMsg) {
      *rv = kv_msg;
      return;
    }
    // Recv ArrayMeta
    Message recv_meta_msg;
    CHECK_EQ(receiver->RecvFrom(&recv_meta_msg, send_id), REMOVE_SUCCESS);
    ArrayMeta meta(recv_meta_msg.data, recv_meta_msg.size);
    recv_meta_msg.deallocator(&recv_meta_msg);
    // Recv ID NDArray
    Message recv_id_msg;
    CHECK_EQ(receiver->RecvFrom(&recv_id_msg, send_id), REMOVE_SUCCESS);
    CHECK_EQ(meta.data_shape_[0], 1);
    kv_msg->id = CreateNDArrayFromRaw(
      {meta.data_shape_[1]},
      DLDataType{kDLInt, 64, 1},
      DLContext{kDLCPU, 0},
      recv_id_msg.data);
    // Recv Data NDArray
    if (kv_msg->msg_type != kPullMsg) {
      Message recv_data_msg;
      CHECK_EQ(receiver->RecvFrom(&recv_data_msg, send_id), REMOVE_SUCCESS);
      CHECK_GE(meta.data_shape_[2], 1);
      std::vector<int64_t> vec_shape;
      for (int i = 3; i < meta.data_shape_.size(); ++i) {
        vec_shape.push_back(meta.data_shape_[i]);
      }
      kv_msg->data = CreateNDArrayFromRaw(
        vec_shape,
        DLDataType{kDLFloat, 32, 1},
        DLContext{kDLCPU, 0},
        recv_data_msg.data);
    }
    *rv = kv_msg;
  });

DGL_REGISTER_GLOBAL("network._CAPI_ReceiverGetKVMsgType")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    KVMsgHandle chandle = args[0];
    network::KVStoreMsg* msg = static_cast<KVStoreMsg*>(chandle);
    *rv = msg->msg_type;
  });

DGL_REGISTER_GLOBAL("network._CAPI_ReceiverGetKVMsgRank")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    KVMsgHandle chandle = args[0];
    network::KVStoreMsg* msg = static_cast<KVStoreMsg*>(chandle);
    *rv = msg->rank;
  });

DGL_REGISTER_GLOBAL("network._CAPI_ReceiverGetKVMsgName")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    KVMsgHandle chandle = args[0];
    network::KVStoreMsg* msg = static_cast<KVStoreMsg*>(chandle);
    *rv = msg->name;
  });

DGL_REGISTER_GLOBAL("network._CAPI_ReceiverGetKVMsgID")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    KVMsgHandle chandle = args[0];
    network::KVStoreMsg* msg = static_cast<KVStoreMsg*>(chandle);
    *rv = msg->id;
  });

DGL_REGISTER_GLOBAL("network._CAPI_ReceiverGetKVMsgData")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    KVMsgHandle chandle = args[0];
    network::KVStoreMsg* msg = static_cast<KVStoreMsg*>(chandle);
    *rv = msg->data;
  });

DGL_REGISTER_GLOBAL("network._CAPI_DeleteKVMsg")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    KVMsgHandle chandle = args[0];
    network::KVStoreMsg* msg = static_cast<KVStoreMsg*>(chandle);
    delete msg;
  });


////////////////////////// Distributed Sampler Components ////////////////////////////////
char* DistSampleMsg::Serialize(int64_t* size) {
  char* buffer = nullptr;
  int64_t buffer_size = 0;
  buffer_size += sizeof(this->msg_type);
  buffer_size += sizeof(this->rank);
  buffer_size += sizeof(this->fanout);

  if (!this->name.empty()) {
    buffer_size += sizeof(this->name.size());
    buffer_size += this->name.size();
  }
  // In the future, we should have a better memory management as
  // allocating a large chunk of memory can be very expensive.
  buffer = new char[buffer_size];
  char* pointer = buffer;
  // write msg_type
  *(reinterpret_cast<int*>(pointer)) = this->msg_type;
  pointer += sizeof(this->msg_type);
  // write rank
  *(reinterpret_cast<int*>(pointer)) = this->rank;
  pointer += sizeof(this->rank);
  // write fanout
  *(reinterpret_cast<int*>(pointer)) = this->fanout;
  pointer += sizeof(this->fanout);
  // write name
  if (!this->name.empty()) {
    *(reinterpret_cast<size_t*>(pointer)) = this->name.size();
    pointer += sizeof(size_t);
    memcpy(pointer, this->name.c_str(), this->name.size());
  }

  *size = buffer_size;
  return buffer;
}

void DistSampleMsg::Deserialize(char* buffer, int64_t size) {
  int64_t data_size = 0;
  // Read msg_type
  this->msg_type = *(reinterpret_cast<int*>(buffer));
  buffer += sizeof(int);
  data_size += sizeof(int);
  // Read rank
  this->rank = *(reinterpret_cast<int*>(buffer));
  buffer += sizeof(int);
  data_size += sizeof(int);

  this->fanout = *(reinterpret_cast<int*>(buffer));
  buffer += sizeof(int);
  data_size += sizeof(int);
  
  if (data_size < size) {
    // Read name
    size_t name_size = *(reinterpret_cast<size_t*>(buffer));
    buffer += sizeof(name_size);
    data_size += sizeof(name_size);
    this->name.assign(buffer, name_size);
    data_size += name_size;
  }
  CHECK_EQ(data_size, size);
}

void DistSampleMsg::Load(dmlc::Stream *stream) {
  stream->Read(&this->msg_type);
  stream->Read(&this->rank);
  stream->Read(&this->fanout);
  stream->Read(&this->name);
}

void _SendDSMsg(network::Sender *sender, const size_t recv_id, DistSampleMsg &ds_msg) {
    int64_t ds_size = 0;
    char* ds_data = ds_msg.Serialize(&ds_size);
    // Send kv_data
    Message send_ds_msg;
    send_ds_msg.data = ds_data;
    send_ds_msg.size = ds_size;
    send_ds_msg.deallocator = DefaultMessageDeleter;
    CHECK_EQ(sender->Send(send_ds_msg, recv_id), ADD_SUCCESS);

    if (ds_msg.msg_type != kFinalMsg &&
        ds_msg.msg_type != kBarrierMsg &&
        ds_msg.msg_type != kIPIDMsg) {
      // Send ArrayMeta
      ArrayMeta meta(ds_msg.msg_type);
      if (ds_msg.msg_type == kSampleRequest) {
        meta.AddArray(ds_msg.seed);
      }
      if (ds_msg.msg_type == kSampleResponse) {
        meta.AddArray(ds_msg.result);
      }
      int64_t meta_size = 0;
      char* meta_data = meta.Serialize(&meta_size);
      Message send_meta_msg;
      send_meta_msg.data = meta_data;
      send_meta_msg.size = meta_size;
      send_meta_msg.deallocator = DefaultMessageDeleter;
      CHECK_EQ(sender->Send(send_meta_msg, recv_id), ADD_SUCCESS);
      // Send seed NDArray
      if (ds_msg.msg_type == kSampleRequest) {
        Message send_seed_msg;
        send_seed_msg.data = static_cast<char*>(ds_msg.seed->data);
        send_seed_msg.size = ds_msg.seed.GetSize();
        NDArray data = ds_msg.seed;
        send_seed_msg.deallocator = [data](Message*) {};
        CHECK_EQ(sender->Send(send_seed_msg, recv_id), ADD_SUCCESS);
      }
      // Send result NDArray
      if (ds_msg.msg_type == kSampleResponse) {
        Message send_result_msg;
        send_result_msg.data = static_cast<char*>(ds_msg.result->data);
        send_result_msg.size = ds_msg.result.GetSize();
        NDArray data = ds_msg.result;
        send_result_msg.deallocator = [data](Message*) {};
        CHECK_EQ(sender->Send(send_result_msg, recv_id), ADD_SUCCESS);
      }
    }
}

DGL_REGISTER_GLOBAL("network._CAPI_SenderSendDSMsg")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    int args_count = 0;
    CommunicatorHandle chandle = args[args_count++];
    int recv_id = args[args_count++];
    DistSampleMsg ds_msg;
    ds_msg.msg_type = args[args_count++];
    ds_msg.rank = args[args_count++];
    ds_msg.fanout = args[args_count++];

    network::Sender* sender = static_cast<network::Sender*>(chandle);
    if (ds_msg.msg_type != kFinalMsg && ds_msg.msg_type != kBarrierMsg) {
      std::string name = args[args_count++];
      ds_msg.name = name;
      if (ds_msg.msg_type == kSampleRequest) {
        ds_msg.seed = args[args_count++];
      }
      if (ds_msg.msg_type == kSampleResponse) {
        ds_msg.result = args[args_count++];
      }
    }
    _SendDSMsg(sender, recv_id, ds_msg);
  });

DistSampleMsg *_RecvDSMsg(network::Receiver* receiver, int send_id = -1){
    DistSampleMsg *ds_msg = new DistSampleMsg();
    // Recv kv_Msg
    Message recv_ds_msg;
    if (send_id  >= 0)
      receiver->RecvFrom(&recv_ds_msg, send_id);
    else
      CHECK_EQ(receiver->Recv(&recv_ds_msg, &send_id), REMOVE_SUCCESS);
    ds_msg->Deserialize(recv_ds_msg.data, recv_ds_msg.size);
    recv_ds_msg.deallocator(&recv_ds_msg);
    if (ds_msg->msg_type == kFinalMsg ||
        ds_msg->msg_type == kBarrierMsg ||
        ds_msg->msg_type == kIPIDMsg) {
      return ds_msg;
    }
    // Recv ArrayMeta
    Message recv_meta_msg;
    CHECK_EQ(receiver->RecvFrom(&recv_meta_msg, send_id), REMOVE_SUCCESS);
    ArrayMeta meta(recv_meta_msg.data, recv_meta_msg.size);
    recv_meta_msg.deallocator(&recv_meta_msg);
    // Recv seed NDArray
    if (ds_msg->msg_type == kSampleRequest) {
      Message recv_seed_msg;
      CHECK_EQ(receiver->RecvFrom(&recv_seed_msg, send_id), REMOVE_SUCCESS);
      CHECK_EQ(meta.data_shape_[0], 1);
      ds_msg->seed = CreateNDArrayFromRaw(
        {meta.data_shape_[1]},
        DLDataType{kDLInt, 64, 1},
        DLContext{kDLCPU, 0},
        recv_seed_msg.data);
    }
    // Recv result NDArray
    if (ds_msg->msg_type == kSampleResponse) {
      Message recv_result_msg;
      CHECK_EQ(receiver->RecvFrom(&recv_result_msg, send_id), REMOVE_SUCCESS);
      CHECK_GE(meta.data_shape_[0], 1);
      std::vector<int64_t> vec_shape;
      for (int i = 1; i < meta.data_shape_.size(); ++i) {
        vec_shape.push_back(meta.data_shape_[i]);
      }
      ds_msg->result = CreateNDArrayFromRaw(
        vec_shape,
        DLDataType{kDLInt, 64, 1},
        DLContext{kDLCPU, 0},
        recv_result_msg.data);
    }
    return ds_msg;
};

DGL_REGISTER_GLOBAL("network.CAPI_ReceiverRecvDSMsg")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CommunicatorHandle chandle = args[0];
    network::Receiver* receiver = static_cast<network::SocketReceiver*>(chandle);
    DistSampleMsg *ds_msg = _RecvDSMsg(receiver);
    *rv = ds_msg;
  });

DGL_REGISTER_GLOBAL("network._CAPI_ReceiverGetDSMsgType")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    DSMsgHandle chandle = args[0];
    network::DistSampleMsg* msg = static_cast<DistSampleMsg*>(chandle);
    *rv = msg->msg_type;
  });

DGL_REGISTER_GLOBAL("network._CAPI_ReceiverGetDSMsgRank")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    DSMsgHandle chandle = args[0];
    network::DistSampleMsg* msg = static_cast<DistSampleMsg*>(chandle);
    *rv = msg->rank;
  });

DGL_REGISTER_GLOBAL("network._CAPI_ReceiverGetDSMsgName")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    DSMsgHandle chandle = args[0];
    network::DistSampleMsg* msg = static_cast<DistSampleMsg*>(chandle);
    *rv = msg->name;
  });

DGL_REGISTER_GLOBAL("network._CAPI_ReceiverGetDSMsgFanout")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    DSMsgHandle chandle = args[0];
    network::DistSampleMsg* msg = static_cast<DistSampleMsg*>(chandle);
    *rv = msg->fanout;
  });

DGL_REGISTER_GLOBAL("network._CAPI_ReceiverGetDSMsgSeed")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    DSMsgHandle chandle = args[0];
    network::DistSampleMsg* msg = static_cast<DistSampleMsg*>(chandle);
    *rv = msg->seed;
  });

DGL_REGISTER_GLOBAL("network._CAPI_ReceiverGetDSMsgResult")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    DSMsgHandle chandle = args[0];
    network::DistSampleMsg* msg = static_cast<DistSampleMsg*>(chandle);
    *rv = msg->result;
  });

DGL_REGISTER_GLOBAL("network._CAPI_DeleteDSMsg")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    DSMsgHandle chandle = args[0];
    network::DistSampleMsg* msg = static_cast<DistSampleMsg*>(chandle);
    delete msg;
  });

void _StreamSendDSMsg(network::Sender *sender, const size_t recv_id, DistSampleMsg &ds_msg) {
    int64_t ds_size = 0;
    BufferStream bs;
    Message send_ds_msg;
    ds_msg.Save(&bs);
    CHECK_EQ(sender->Send(bs.ToMessage(), recv_id), ADD_SUCCESS);

    if (ds_msg.msg_type != kFinalMsg &&
        ds_msg.msg_type != kBarrierMsg &&
        ds_msg.msg_type != kIPIDMsg) {
      // Send ArrayMeta
      ArrayMeta meta(ds_msg.msg_type);
      if (ds_msg.msg_type == kSampleRequest) {
        meta.AddArray(ds_msg.seed);
      }
      if (ds_msg.msg_type == kSampleResponse) {
        meta.AddArray(ds_msg.result);
      }
      bs.Reset();
      meta.Save(&bs);
      CHECK_EQ(sender->Send(bs.ToMessage(), recv_id), ADD_SUCCESS);
      // Send seed NDArray
      if (ds_msg.msg_type == kSampleRequest) {
        CHECK_EQ(sender->Send(Message(ds_msg.seed), recv_id), ADD_SUCCESS);
      }
      // Send result NDArray
      if (ds_msg.msg_type == kSampleResponse) {
        CHECK_EQ(sender->Send(Message(ds_msg.result), recv_id), ADD_SUCCESS);
      }
    }
}

DistSampleMsg *_StreamRecvDSMsg(network::Receiver* receiver) {
    DistSampleMsg *ds_msg = new DistSampleMsg();
    // Recv kv_Msg
    Message recv_ds_msg;
    int send_id;
    InputBufferStream is;
    CHECK_EQ(receiver->Recv(&recv_ds_msg, &send_id), REMOVE_SUCCESS);
    is.Reset(recv_ds_msg.data);
    recv_ds_msg.deallocator(&recv_ds_msg);

    ds_msg->Load(&is);
    // Recv ArrayMeta
    Message recv_meta_msg;
    CHECK_EQ(receiver->RecvFrom(&recv_meta_msg, send_id), REMOVE_SUCCESS);
    is.Reset(recv_meta_msg.data);
    ArrayMeta meta(&is);
    recv_meta_msg.deallocator(&recv_meta_msg);

    // Recv seed NDArray
    if (ds_msg->msg_type == kSampleRequest) {
      Message recv_seed_msg;
      CHECK_EQ(receiver->RecvFrom(&recv_seed_msg, send_id), REMOVE_SUCCESS);
      CHECK_EQ(meta.data_shape_[0], 1);
      ds_msg->seed = CreateNDArrayFromRaw(
        {meta.data_shape_[1]},
        DLDataType{kDLInt, 64, 1},
        DLContext{kDLCPU, 0},
        recv_seed_msg.data);
    }
    
    // Recv result NDArray
    if (ds_msg->msg_type == kSampleResponse) {
      Message recv_result_msg;
      CHECK_EQ(receiver->RecvFrom(&recv_result_msg, send_id), REMOVE_SUCCESS);
      CHECK_GE(meta.data_shape_[0], 1);
      std::vector<int64_t> vec_shape;
      for (int i = 1; i < meta.data_shape_.size(); ++i) {
        vec_shape.push_back(meta.data_shape_[i]);
      }
      ds_msg->result = CreateNDArrayFromRaw(
        vec_shape,
        DLDataType{kDLInt, 64, 1},
        DLContext{kDLCPU, 0},
        recv_result_msg.data);
    }
    return ds_msg;
};

void _SendSampleRequest(network::Sender* sender,const size_t recv_id, 
                        const std::vector<dgl_id_t> &seeds, int fanout, const size_t client_id) {
    DistSampleMsg ds_msg;
    ds_msg.msg_type = kSampleRequest;
    ds_msg.rank = client_id;
    ds_msg.fanout = fanout;
    ds_msg.name = " ";
    // Is this conversion necessary?
    ds_msg.seed = NDArray::FromVector(seeds);
    _SendDSMsg(sender, recv_id, ds_msg);
}

using HeterpSubGraphPtr = std::shared_ptr<HeteroSubgraph>;

#include <chrono>
static double t_union = 0;
static double t_req_wait = 0;
static double t_g2l = 0;
static double t_sample = 0;
static double t_l2g = 0;
static double t_rebuild = 0;
static double t_serial = 0;

HeterpSubGraphPtr HeteroSubgraphUnion(const std::vector<HeterpSubGraphPtr> &components) {
  auto start = std::chrono::steady_clock::now();
  HeteroGraphPtr graph = components[0]->graph;
  GraphPtr meta_graph = graph->meta_graph();
  // concat COO
  std::vector<HeteroGraphPtr> rel_graphs;
  std::vector<std::vector<int64_t>> induced_edges(graph->NumEdgeTypes()); 
  for (dgl_type_t etype = 0; etype < graph->NumEdgeTypes(); etype++){
    auto rel = components[0]->graph->GetRelationGraph(etype);
    int64_t coo_size = 0;
    std::vector<int64_t> coo_size_list;
    for (const auto &comp : components) {
      coo_size += comp->graph->GetCOOMatrix(etype).col->shape[0];
      coo_size_list.push_back(coo_size);
    }
    std::vector<int64_t> rarr;
    std::vector<int64_t> carr;
    rarr.reserve(coo_size); carr.reserve(coo_size);
    IdArray darr = aten::NullArray();
    auto nrows = rel->GetCOOMatrix(0).num_rows, ncols = rel->GetCOOMatrix(0).num_cols;
    auto &ie = induced_edges[etype];
    for (const auto &comp : components) {
      const auto &comp_coo = comp->graph->GetCOOMatrix(etype);
      const int64_t *v_row = static_cast<const int64_t *>(comp_coo.row->data);
      const int64_t *v_col = static_cast<const int64_t *>(comp_coo.col->data);

      rarr.insert(rarr.end(), v_row, v_row + comp_coo.row->shape[0]);
      carr.insert(carr.end(), v_col, v_col + comp_coo.col->shape[0]);

      const auto &ie_cur_type = comp->induced_edges[etype];
      auto comp_edges = static_cast<const int64_t *>(ie_cur_type->data);
      ie.insert(ie.begin(), comp_edges, comp_edges + ie_cur_type->shape[0]);
    } 
    rel_graphs.emplace_back(UnitGraph::CreateFromCOO(
      rel->NumVertexTypes(),
      nrows,
      ncols,
      aten::VecToIdArray(rarr),
      aten::VecToIdArray(carr)));
  }
  HeteroGraphPtr union_graph = std::make_shared<HeteroGraph>(
    meta_graph,
    rel_graphs
  );
  auto union_subgraph = std::make_shared<HeteroSubgraph>();
  union_subgraph->graph = union_graph;
  for (dgl_type_t etype = 0; etype < graph->NumEdgeTypes(); etype++)
    union_subgraph->induced_edges.push_back(aten::VecToIdArray(induced_edges[etype]));
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> t = std::chrono::duration<double>(end - start);
  t_union += t.count();
  return union_subgraph;
}

static double t_remote = 0;
static double t_split = 0;
// For Distributed sampling
DGL_REGISTER_GLOBAL("network._CAPI_RemoteSamplingReqeust")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  CommunicatorHandle rchandle = args[0];
  network::Receiver* receiver = static_cast<network::Receiver*>(rchandle);
  CommunicatorHandle schandle = args[1];
  network::Sender* sender = static_cast<network::Sender*>(schandle);

  const int client_id = args[2];
  const int num_parts = args[3];
  NDArray part_ids = args[4];
  IdArray seed_nodes = args[5];
  int64_t fanout = args[6];
  
  std::vector<dgl_id_t> *part_req = new std::vector<dgl_id_t>[num_parts];
  auto start = std::chrono::steady_clock::now();
  const int64_t seed_size = seed_nodes->shape[0];
  const dgl_id_t *seed_arr = static_cast<dgl_id_t *>(seed_nodes->data);
  const int64_t *part_id_arr = static_cast<int64_t *>(part_ids->data);
  for (size_t i = 0 ; i < seed_size; ++i) {
    const dgl_id_t node_id = seed_arr[i];
    const int64_t part_id = part_id_arr[node_id];
    part_req[part_id].emplace_back(node_id);
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> t = std::chrono::duration<double>(end - start);
  t_split += t.count();

  // TODO: Use OpenMP to optimize
  size_t req_tot = 0;
  start = std::chrono::steady_clock::now();
  for (size_t i = 0 ; i < num_parts; i++)
    if (!part_req[i].empty()) {
      _SendSampleRequest(sender, i, part_req[i], fanout, client_id);
      ++req_tot;
    }

  std::vector<HeterpSubGraphPtr> responses;
  responses.reserve(req_tot);
  for (size_t i = 0 ; i < req_tot ; ++i) {
      Message msg;
      int send_id;
      CHECK_EQ(receiver->Recv(&msg, &send_id), REMOVE_SUCCESS);
      InputBufferStream is(msg.data);
      auto subgraph = std::make_shared<HeteroSubgraph>();
      subgraph->Load(&is);
      responses.emplace_back(subgraph);
  }

  end = std::chrono::steady_clock::now();
  t = std::chrono::duration<double>(end - start);
  t_req_wait += t.count();
  
  start = std::chrono::steady_clock::now();
  // union
  auto union_graph = HeteroSubgraphUnion(responses);
  end = std::chrono::steady_clock::now();
  t = std::chrono::duration<double>(end - start);
  t_union += t.count();

  *rv = HeteroSubgraphRef(union_graph);
  std::cout << "t_split: " << t_split << " | t_wait: " << t_req_wait << " | t_union: " << t_union << std::endl; 

  delete []part_req;
});


static double t_l2g1 = 0;
static double t_l2g2 = 0;

void ServerLoop(
  size_t thread_id,
  network::Receiver* receiver,
  network::Sender* sender,
  int64_t num_vertices,
  IdArray g2l,
  IdArray l2g,
  IdArray e_l2g,
  HeteroGraphRef hg,
  const std::vector<FloatArray> &prob
) {
  while (true) {
    DistSampleMsg *msg;
    msg = _RecvDSMsg(receiver, thread_id);
    if (msg->msg_type == kFinalMsg) {
      std::cout << "Service stops." << std::endl;
      return ;
    }

    std::vector<int64_t> fanouts;
    fanouts.emplace_back(msg->fanout);

    std::vector<IdArray> nodes;

    auto start = std::chrono::steady_clock::now();
    // convert global id into local id
    nodes.emplace_back(aten::IndexSelect(g2l, msg->seed));

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> t = std::chrono::duration<double>(end - start);
    t_g2l += t.count();

    start = std::chrono::steady_clock::now();
    std::shared_ptr<HeteroSubgraph> subg(new HeteroSubgraph);
    *subg = dgl::sampling::SampleNeighbors(
      hg.sptr(), nodes, fanouts, EdgeDir::kIn, prob, true
    );
    end = std::chrono::steady_clock::now();
    t = std::chrono::duration<double>(end - start);
    t_sample += t.count();

    start = std::chrono::steady_clock::now();

    // convert local id to global id
    std::vector<HeteroGraphPtr> rel_graphs;
    for (dgl_type_t etype = 0; etype < subg->graph->NumEdgeTypes(); ++etype) {
      auto coo = subg->graph->GetCOOMatrix(etype);
      // #pragma parallel for
      IdArray row = aten::IndexSelect(l2g, coo.row);
      IdArray col = aten::IndexSelect(l2g, coo.col);

      auto rel = std::dynamic_pointer_cast<HeteroGraph>(subg->graph);
      rel_graphs.emplace_back(UnitGraph::CreateFromCOO(
        rel->NumVertexTypes(),
        num_vertices,
        num_vertices,
        row,
        col
      ));
    }
    end = std::chrono::steady_clock::now();
    t = std::chrono::duration<double>(end - start);
    t_l2g1 += t.count();

    std::vector<IdArray> induced_edges;
    for (auto &ie : subg->induced_edges) {
      induced_edges.emplace_back(aten::IndexSelect(e_l2g, ie));
    }

    end = std::chrono::steady_clock::now();
    t = std::chrono::duration<double>(end - start);
    t_l2g2 += t.count();

    auto relabled_graph = std::make_shared<HeteroGraph>(
      subg->graph->meta_graph(),
      rel_graphs
    );

    end = std::chrono::steady_clock::now();
    t = std::chrono::duration<double>(end - start);
    t_l2g += t.count();

    start = std::chrono::steady_clock::now();
    auto relabled_subgraph = std::make_shared<HeteroSubgraph>();
    relabled_subgraph->graph = relabled_graph;
    relabled_subgraph->induced_edges = induced_edges;
    relabled_subgraph->induced_vertices = std::vector<IdArray>();
    BufferStream bs;
    relabled_subgraph->Save(&bs);
    end = std::chrono::steady_clock::now();
    t = std::chrono::duration<double>(end - start);
    t_serial += t.count();
    Message response = bs.ToMessage();
    sender->Send(response, msg->rank);
    // uint64_t a = *reinterpret_cast<uint64_t *>(response.data);
    // std::cout << "subgraph sent back, size:" << response.size << " magic:" << a << std::endl;

    std::cout << "## used time ##" << std::endl;
    std::cout << "g2l: " << t_g2l << " | sample: " << t_sample << " | l2g: " << t_l2g  
              << " (stage1: "<<t_l2g1 << ", stage2:" << t_l2g2 << ")"
              << " | serialization: " << t_serial << std::endl;
  }
}

DGL_REGISTER_GLOBAL("network._CAPI_RemoteSamplingServerLoop")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  int64_t num_client = args[0];
  CommunicatorHandle rchandle = args[1];
  network::Receiver* receiver = static_cast<network::Receiver*>(rchandle);
  CommunicatorHandle schandle = args[2];
  network::Sender* sender = static_cast<network::Sender*>(schandle);
  int64_t num_vertices = args[3];
  IdArray global2local = args[4];
  IdArray local2global = args[5];
  IdArray edge_local2global = args[6];
  HeteroGraphRef hg = args[7];
  const auto& prob = ListValueToVector<FloatArray>(args[8]);

  std::vector<std::thread> threads;
  for (size_t tid = 0 ; tid < num_client ; ++tid){
    threads.emplace_back(ServerLoop, tid, receiver, sender, num_vertices, global2local, local2global, edge_local2global, hg, prob);
  }

  for (auto &thread : threads)
    thread.join();
});

void DistSampleMsg::Save(dmlc::Stream *stream) const {
  stream->Write(this->msg_type);
  stream->Write(this->rank);
  stream->Write(this->fanout);
  stream->Write(this->name);

  stream->Write(this->result);
  stream->Write(this->seed);
}




}  // namespace network
}  // namespace dgl
