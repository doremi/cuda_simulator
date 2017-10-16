#pragma once

#include <cassert>
#include <ctime>
#include <string>
#include <vector>
#include <map>
#include <mutex>

template <typename T>
class SharedMemory {
private:
    static std::vector<T> cache;
    static std::vector<T> persistence;
    size_t _size;

    // 寫入是放進 cache
    // 讀取是從 persistence 讀出
    // 直到呼叫 __syncthreads 才會把 cache 寫入 persistence
    class Proxy {
    public:
        Proxy(size_t idx, size_t start_loc) : idx(idx + start_loc) {}

        // 模擬有一定的機率會讀到舊資料
        // 例如有 A,B 兩個 thread, A 寫入 SharedMemory[0] = 10;
        // 則 B 讀取 SharedMemory[0] 前, 如果沒有 __syncthreads(),
        // 則有一半的機率是讀到舊的資料, 一半機率讀到新資料
        T& operator= (int x) {
            SharedMemory<T>::cache[idx] = x;
            if ((random() % 2) == 0) {
                SharedMemory<T>::persistence[idx] = x;
            }
            return SharedMemory<T>::cache[idx];
        }

        operator T() const {
            return SharedMemory<T>::persistence[idx];
        }

    private:
        size_t idx;
    };

    static size_t total_size;
    size_t start_loc;
    std::map<std::string, size_t> location_table;
    //static std::mutex mtx;

public:
    // 相同名字 (例如在 thread 裡面建立的) 則 start_loc 應該要一樣
    SharedMemory(const size_t size, std::string symbol_name) {
        //std::lock_guard<std::mutex> lock(SharedMemory<T>::mtx);
        auto iter = location_table.find(symbol_name);
        if (iter != location_table.end()) {
            start_loc = iter->second;
        } else {
            start_loc = SharedMemory<T>::total_size;
            location_table[symbol_name] = start_loc;
            SharedMemory<T>::total_size += size;
        }
        _size = size;
        srandom(time(nullptr));
        printf("total size: %lu\n", SharedMemory<T>::total_size);
    }

    ~SharedMemory() {
//        delete [] cache;
//        delete [] persistence;
    }

    static void __syncthreads() {
        SharedMemory<T>::persistence = SharedMemory<T>::cache;
    }

    Proxy operator[](size_t index) {
        assert(index >=0 && index < _size);
        return Proxy(index, start_loc);
    }

    size_t size() const {
        return _size;
    }
};

// 如果可以做到  __shared(int x, 3) 更好
#define __shared(Type, Name, Size) SharedMemory<Type> Name(Size, #Name)
//#define __shared(variable) ((decltype(&variable))nullptr, #variable)
//template<typename T> using __shared = SharedMemory<T>;