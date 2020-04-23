#ifndef STREAM_H
#define STREAM_H
#include <dmlc/io.h>
#include "network.h"

namespace dgl {
class BufferStream : public dmlc::SeekStream {
 public:
  using dmlc::SeekStream::Read;
  using dmlc::SeekStream::Write;
  BufferStream(size_t init_capacity = 256);
  virtual size_t Read(void *ptr, size_t size) override;
  virtual void Write(const void *ptr, size_t size) override;
  virtual void Seek(size_t pos) override;
  virtual size_t Tell(void) override;
  virtual ~BufferStream(void);

  const char* GetBuffer(void) const;
  void Reset(size_t init_capacity = 256);
  network::Message ToMessage();
  
 private:
  size_t capacity;
  char *buffer;
  char *ptr;
  void Resize(const size_t new_size);
  bool converted = false;
};

class InputBufferStream : public dmlc::SeekStream {
 public:
  using dmlc::SeekStream::Read;
  InputBufferStream(const char *buffer=nullptr) : buffer(buffer), ptr(buffer) {}
  virtual size_t Read(void *ptr, size_t size) override;
  virtual void Write(const void *ptr, size_t size) override {
    // throw exception
  }
  virtual void Seek(size_t pos) override;
  virtual size_t Tell(void) override;
  
  void Reset(const char *buffer) {
    this->buffer = this->ptr = buffer;
  }

  virtual ~InputBufferStream(void) {}
 private:
  const char *buffer;
  const char *ptr;
};

}
#endif