#include "stream.h"
#include <dgl/runtime/object.h>

namespace dgl{
BufferStream::BufferStream(size_t init_capacity) : capacity(init_capacity) {
  this->ptr = this->buffer = static_cast<char *>(malloc(init_capacity * sizeof(char)));
}

size_t BufferStream::Read(void *ptr, size_t size) {
  if (this->converted) return 0;
  size_t read_size = std::min(size, this->capacity - (this->ptr - this->buffer));
  memcpy(ptr, this->ptr, read_size);
  this->ptr += read_size;
  return read_size;
} 

void BufferStream::Write(const void *ptr, size_t size) {
  if (this->converted) return;
  // std::cout << "write " << size << std::endl;
  size_t cap = this->capacity;
  while (size + this->ptr > this->buffer + cap) cap *= 2;
  if (cap > this->capacity) Resize(cap);
  // std::cerr << this->capacity - (this->ptr - this->buffer) << " " << size << std::endl;
  memcpy(this->ptr, ptr, size);
  this->ptr += size;
}

void BufferStream::Seek(size_t pos) {
  if (this->converted) return;
  size_t cap = this->capacity;
  while (pos >= cap) cap*=2;
  if (cap > this->capacity) Resize(cap);
  this->ptr = buffer + pos;
}

size_t BufferStream::Tell(void){
  return this->ptr - this->buffer;
}


void BufferStream::Resize(size_t new_size) {
  if (this->converted) return;
// std::cout << "buffer resize" << std::endl;
  int offset = this->ptr - this->buffer;
  CHECK(offset >= 0);
  this->capacity = new_size;
  this->buffer = static_cast<char*>(realloc(this->buffer, this->capacity));
  this->ptr = this->buffer + offset;
}

const char* BufferStream::GetBuffer(void) const {
  return this->buffer;
}

void BufferStream::Reset(size_t init_capacity) {
  if (this->buffer && !this->converted)
    delete [] this->buffer;
  this->ptr = this->buffer = static_cast<char *>(malloc(init_capacity * sizeof(char)));
  this->capacity = init_capacity;
}

BufferStream::~BufferStream(void) {
  if (this->buffer && !this->converted)
    delete [] this->buffer;
  this->buffer = nullptr;
}

network::Message BufferStream::ToMessage() {
  this->converted = true;
  network::Message msg(buffer, ptr - buffer);
  msg.deallocator = network::DefaultMessageDeleter;
  return msg;
}

size_t InputBufferStream::Read(void *ptr, size_t size) {
  memcpy(ptr, this->ptr, size);
  this->ptr += size;
  return size;
}

void InputBufferStream::Seek(size_t pos) {
  this->ptr = this->buffer + pos;
}

size_t InputBufferStream::Tell(void) {
  return this->ptr - this->buffer;
}

}