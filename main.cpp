#include <iostream>
#include <thread>
#include "Barrier.hpp"

struct uint3 {
    unsigned int x, y, z;
};

struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) noexcept : x(vx), y(vy), z(vz) {}
    explicit dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
    explicit operator uint3() { uint3 t; t.x = x; t.y = y; t.z = z; return t; }
};

thread_local uint3 threadIdx;
thread_local uint3 blockIdx;
dim3 blockDim;
dim3 gridDim;
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

void radix2(const int * const d_input1,
           const int * const d_input2,
           int * const d_output)
{
    int arrayLength = 16;
    int stride = 1;
    int global_comparatorI = threadIdx.x;
    int comparatorI = global_comparatorI & (arrayLength / 2 - 1);
    int pos = 2 * global_comparatorI - (global_comparatorI & (stride - 1));
    __syncthreads();
    printf("[thread %d], %d - %d\n", threadIdx.x, pos, pos + stride);
}
struct KernelData {
    unsigned int tid;
    unsigned int block_id;
    int *d_input1;
    int *d_input2;
    int *d_output;
};

template<int N>
void my_kernel(int *d_output)
{
    int tid = threadIdx.x;
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ int mem[3] = {0};
    ++mem[tid];
    __syncthreads();
    d_output[index] = mem[tid];
}

void prepare_kernel(const struct KernelData *kd)
{
    blockIdx.x = kd->block_id;
    threadIdx.x = kd->tid;
    my_kernel<3>(kd->d_output);
    //kernel(kd->d_input, kd->d_output);
    //radix2(kd->d_input1, kd->d_input2, kd->d_output);
}
#endif

// 先支援一維
void start_kernel(dim3 blocks, dim3 threads)
{
    // 不同 blocks 的 __shared__ 不能互通
    // eg: blocks.x = 2, threads.x = 3
    // total: blocks.x * threads.x = 6
    const size_t size = blocks.x * threads.x;
    auto *t = new std::thread[size];
    auto d_output = new int[size];
    gridDim.x = blocks.x;
    blockDim.x = threads.x;
    barrier = new Barrier(size); // 應該要是每個 block 各自的, 現在先用 global lock
    struct KernelData kd[size];
    for (auto i = 0; i < blocks.x; ++i) {
        for (auto j = 0; j < threads.x; ++j) {
            auto n = i * threads.x + j;
            kd[n] = {
                    .block_id = static_cast<unsigned int>(i),
                    .tid = static_cast<unsigned int>(j),
                    .d_output = d_output,
            };
            t[n] = std::thread(prepare_kernel, &kd[n]);
        }
    }

    for (auto i = 0; i < size; ++i) {
        t[i].join();
    }

    for (auto i = 0; i < size; ++i) {
        printf("output[%d] = %d\n", i, d_output[i]);
    }
}

int main()
{
    start_kernel(2, 3);
    return 0;
}

#if 0
// 接下來可做的：
// 1. 二維 block
// 2. 解決尾巴 block 不夠 512 threads 要怎麼排序
// 3. 把 radix 移植過來
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

    for (auto i = 0; i < size; ++i) {
        kd[i] = {.tid = static_cast<unsigned int>(i), .d_input1 = d_input1, .d_input2 = d_input2, .d_output = d_output};
        t[i] = std::thread(prepare_kernel, &kd[i]);
    }

    for (auto &i : t) {
        i.join();
    }

//    for (auto i = 0; i < size * 2; ++i) {
//        printf("output[%d] = %d\n", i, d_output[i]);
//    }

    return 0;
}
#endif