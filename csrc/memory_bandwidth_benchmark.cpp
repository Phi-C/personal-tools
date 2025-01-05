// Copyright 2024 AI Inc. All rights reserved.
// Author: Chen Chen (chenxj@163.com)
//
// Memory bandwidth benchmark
// Usage:
//  1. Compile: gcc -o memory_bandwidth_benchmark memory_bandwidth_benchmark.cpp
//              gcc -O0 memory_bandwidth_benchmark
//              memory_bandwidth_benchmark.cpp [disable compile optimization]
//  2. Run: ./memory_bandwidth_benchmark

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define ARRAY_SIZE (1024L * 1024L * 1024L) // 1 GB
#define REPEAT_COUNT 10                    // Repeat the test multiple times

// 获取当前时间（以秒为单位）
double get_time_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

// 顺序访问内存
void sequential_access(float *arr, size_t size) {
    for (size_t i = 0; i < size; i++) {
        arr[i] = arr[i] * 1.0f; // 简单的计算，防止优化器优化掉访问
    }
}

// 随机访问内存
void random_access(float *arr, size_t size) {
    for (size_t i = 0; i < size; i++) {
        size_t index = rand() % size;
        arr[index] = arr[index] * 1.0f; // 同样做一个简单的计算
    }
}

// 主程序
int main() {
    // 分配内存
    float *arr = (float *)malloc(ARRAY_SIZE * sizeof(float));
    if (!arr) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // 测试顺序访问
    double start_time = get_time_seconds();
    for (int i = 0; i < REPEAT_COUNT; i++) {
        sequential_access(arr, ARRAY_SIZE);
    }
    double end_time = get_time_seconds();
    printf("Sequential access time: %.6f seconds\n", end_time - start_time);
    printf("Bandwidth: %.2f GB/s\n",
           (ARRAY_SIZE * sizeof(float) * REPEAT_COUNT) /
               (end_time - start_time) / 1e9);

    // 测试随机访问
    start_time = get_time_seconds();
    for (int i = 0; i < REPEAT_COUNT; i++) {
        random_access(arr, ARRAY_SIZE);
    }
    end_time = get_time_seconds();
    printf("Random access time: %.6f seconds\n", end_time - start_time);
    printf("Bandwidth: %.2f GB/s\n",
           (ARRAY_SIZE * sizeof(float) * REPEAT_COUNT) /
               (end_time - start_time) / 1e9);

    // 释放内存
    free(arr);
    return 0;
}
