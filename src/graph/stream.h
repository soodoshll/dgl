#ifndef STREAM_H
#define STREAM_H
#include <dmlc/io.h>

namespace dgl {
class BufferStream : dmlc::SeekStream {
 public:
  BufferStream(size_t init_capacity = 256);
  virtual size_t Read(void *ptr, size_t size);
  virtual void Write(const void *ptr, size_t size);
  virtual void Seek(size_t pos);
  virtual size_t Tell(void);
  virtual ~BufferStream(void) {}

 private:
  size_t capacity;
  char *buffer;
  char *ptr;
  void Resize(const size_t new_size);
};

}
#endif