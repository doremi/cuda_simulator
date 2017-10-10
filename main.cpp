#include <iostream>
#include <thread>
#include "Barrier.hpp"

struct uint3 {
    unsigned int x, y, z;
};

struct dim3 {
    unsigned int x, y, z;
    explicit dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) noexcept : x(vx), y(vy), z(vz) {}
    explicit dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
    explicit operator uint3() { uint3 t; t.x = x; t.y = y; t.z = z; return t; }
};

thread_local uint3 threadIdx;
dim3 blockDim;

Barrier *barrier = nullptr;
std::once_flag sync_flag;

void __syncthreads()
{
    barrier->wait();
//    std::call_once(sync_flag, []() {
//        SharedMemory<int>::__syncthreads();
//    });
}

#if 1

// atomic_inc 這個應該用 global lock 就可以暫時解決了
//#define __shared__ __attribute__ ((section ("DATA,THREADS"))) static
#define __shared__ static

void kernel(const int * const d_input,
            int * const d_output)
{
    int tid = threadIdx.x;
    __shared__ int mem[4];
    mem[tid] = d_input[tid];

    printf("thread %d, [%d %d %d %d]\n", tid, mem[0], mem[1], mem[2], mem[3]);
    __syncthreads();
    printf("thread %d, [%d %d %d %d]\n", tid, mem[0], mem[1], mem[2], mem[3]);

    int sum = 0;
    for (auto i = 0; i < blockDim.x; ++i) {
        sum += mem[i];
    }

    d_output[tid] = sum;
}

// 先假設有 8 threads, input 兩組, 每組 8 個數字, output 為 16 個數字
void radix(const int * const d_input1,
           const int * const d_input2,
           int * const d_output)
{
    int tid = threadIdx.x;
    int swap_offset = 8;
//    int count = 1;
    int x = 0, y = 0;

    __shared__ int x_array[16];
    __shared__ int y_array[16];
    __shared__ int buff[16];

    x_array[tid * 2] = y_array[tid * 2] = tid * 2;
    x_array[tid * 2 + 1] = y_array[tid * 2 + 1] = tid * 2 + 1;

    buff[tid] = d_input1[tid];
    buff[blockDim.x + tid] = d_input2[blockDim.x - tid - 1];

    for (int i = blockDim.x; i > 0; i >>= 1) {
//        if (tid == 0) {
//            printf("第 %d 回合, swap_offset = %d\n==========\n", count++, swap_offset);
//        }
        __syncthreads();

        // 每回合要讀取 x_array, y_array 前先讀入 local storage
        int local_x[16], local_y[16];
        for (int cp = 0; cp < 16; ++cp) {
            local_x[cp] = x_array[cp];
            local_y[cp] = y_array[cp];
        }
        __syncthreads();

        // 不管做幾回合, 都是要比較8次
        if (((tid / swap_offset) % 2) == 0) { // 商是偶數, x 按照 tid, y 換成上一步的 x 下半部
            x = local_x[tid];
            y = local_x[tid + swap_offset];
        } else { // 基數, x 換成上一步的 y 上半部, y 按照 tid
            x = local_y[tid - swap_offset];
            y = local_y[tid];
        }
        x_array[tid] = x;
        y_array[tid] = y;
        //printf("%d - %d", x, y);

        //開始比較
        if (buff[x] > buff[y]) {
            auto tmp = buff[x];
            buff[x] = buff[y];
            buff[y] = tmp;
        }

        swap_offset >>= 1;

        __syncthreads();
//        if (tid == 7) {
//            printf("----------\n");
//        }
        __syncthreads();
    }

    __syncthreads();
    d_output[tid * 2] = buff[tid * 2];
    d_output[tid * 2 + 1] = buff[tid * 2 + 1];
    __syncthreads();
}

struct KernelData {
    unsigned int tid;
    int *d_input1;
    int *d_input2;
    int *d_output;
};

void prepare_kernel(const struct KernelData *kd)
{
    threadIdx.x = kd->tid;
    //kernel(kd->d_input, kd->d_output);
    radix(kd->d_input1, kd->d_input2, kd->d_output);
}
#endif

#if 1
// 目標：開 thread, 設定 blockDim
int main()
{
    const size_t size = 8;
    std::thread t[size];
//    int d_input1[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
//    int d_input2[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
//    int d_input1[4] = {1, 2, 4, 9};
//    int d_input2[4] = {3, 4, 7, 20};
    int d_input1[8] = {1, 2, 4, 7, 8, 9, 20, 22};
    int d_input2[8] = {2, 5, 10, 10, 15, 18, 23, 24};
    int d_output[size * 2] = {0};
    struct KernelData kd[size];

    blockDim.x = size;

    barrier = new Barrier(size);
//    for (auto i = 0; i < size; ++i) {
//        d_input[i] = i;
//    }

    for (auto i = 0; i < size; ++i) {
        kd[i] = {.tid = static_cast<unsigned int>(i), .d_input1 = d_input1, .d_input2 = d_input2, .d_output = d_output};
        t[i] = std::thread(prepare_kernel, &kd[i]);
    }

    for (auto &i : t) {
        i.join();
    }

    for (auto i = 0; i < size * 2; ++i) {
        printf("output[%d] = %d\n", i, d_output[i]);
    }

//    delete [] d_input;
//    delete [] d_output;

    return 0;
}
#endif