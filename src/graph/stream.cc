#include "stream.h"
#include <dgl/runtime/object.h>

namespace dgl{
BufferStream::BufferStream(size_t init_capacity = 256) : capacity(capacity) {
  this->ptr = this->buffer = static_cast<char *>(malloc(init_capacity * sizeof(char)));
}

size_t BufferStream::Read(void *ptr, size_t size) {
  size_t read_size = std::min(size, this->capacity - (this->ptr - this->buffer));
  memcpy(ptr, this->ptr, read_size);
  this->ptr += read_size;
  return read_size;
} 

void BufferStream::Write(const void *ptr, size_t size) {
  size_t cap = this->capacity;
  while (size + this->ptr > this->buffer + cap) cap *= 2;
  if (cap > this->capacity) Resize(cap);
  memcpy(this->ptr, ptr, size);
}

void BufferStream::Seek(size_t pos) {
  size_t cap = this->capacity;
  while (pos >= cap) cap*=2;
  if (cap > this->capacity) Resize(cap);
  this->ptr = buffer + pos;
}

size_t BufferStream::Tell(void) {
  return this->ptr - this->buffer;
}


void BufferStream::Resize(size_t new_size) {
  int offset = this->ptr - this->buffer;
  CHECK(offset >= 0);
  this->capacity = new_size;
  this->buffer = static_cast<char*>(realloc(this->buffer, this->capacity));
  this->ptr = this->buffer + offset;
}
}