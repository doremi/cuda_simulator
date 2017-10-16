#include "Shared.hpp"

template<> size_t SharedMemory<int>::total_size = 0;
template<> size_t SharedMemory<unsigned int>::total_size = 0;

template<> std::vector<int> SharedMemory<int>::cache(1024);
template<> std::vector<int> SharedMemory<int>::persistence(1024);

//要在 sharedmemoery constructor 加 lock, 但下面這一行一直無法 build
//template<> std::mutex SharedMemory<int>::mtx;