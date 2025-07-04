---
title: "CUDA Programming Notes | 01: Memory Coalescing"
date: 2025-03-16T01:39:00+08:00
lastmod: 2025-05-26T01:39:00+08:00
draft: false
author: ["jamesnulliu"]
keywords: 
    - memory coalescing
categories:
    - notes 
tags:
    - cuda
    - optimization
description: Introduction to memory coalescing with Nsight Compute.
summary: Introduction to memory coalescing with Nsight Compute.
comments: true
images:
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---

## 1. Introduction to Memory Coalescing

### 1.1. Dynamic Random Access Memories (DRAMs)

Accessing data in the global memory is critical to the performance of a CUDA application.

In addition to tiling techniques utilizing shared memories, we discuss memory coalescing techniques to move data efficiently **from global memory into shared memory and registers**.

Global memory is implemented with dynamic random access memories (DRAMs). Reading one DRAM is a very slow process.  

Modern DRAMs use a parallel process: **Each time a location is accessed, many consecutive locations that includes the requested location are accessed**.

If an application uses data from consecutive locations before moving on to other locations, the DRAMs work close to the advertised peak global memory bandwidth.

### 1.2. Memory Coalescing

Recall that **all threads in a warp execute the same instruction**.

When all threads in a warp execute a load instruction, the hardware detects whether the threads access consecutive memory locations.

The most favorable global memory access is achieved when the same instruction for all threads in a warp accesses global memory locations.

In this favorable case, the hardware coalesces all memory accesses into a consolidated access to consecutive DRAM locations.

> **Definition: Memory Coalescing**  
> If, in a warp, thread $0$ accesses location $n$, thread $1$ accesses location $n + 1$, ... thread $31$ accesses location $n + 31$, then all these accesses are coalesced, that is: **combined into one single access**.

The CUDA C Best Practices Guide gives a high priority recommendation to coalesced access to global memory.

## 2. Example: Vector Addition

### 2.1. Coalesced Access

 
**Coalesced Memory Access** means that each thread in a warp accesses consecutive memory locations so that the hardware can combine all these accesses into one single access. By doing so, fewer wasted data are transferred and the memory bandwidth is fully utilized.

{{< details title="Click to See Example Code" >}}
```cpp {linenos=true}
__global__ void vecAddKernel(const fp32_t* a, const fp32_t* b, fp32_t* c,
                             int32_t n)
{

    int gtid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gtid < n) {
        // [DRAM] 2 load, 1 store, 3 inst
        c[gtid] = a[gtid] + b[gtid];
    }
}

void launchVecAdd(const fp32_t* d_A, const fp32_t* d_B, fp32_t* d_C, size_t n)
{
    dim3 blockSize = {std::min<uint32_t>(n, 1024), 1, 1};
    dim3 gridSize = {ceilDiv<uint32_t>(n, blockSize.x), 1, 1};

    vecAddKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, int32_t(n));
}
```
{{< /details >}}

Note that in NVIDIA GPUs:  
- **WARP** is the smallest unit of execution, which contains 32 threads.
- **SECTOR** is the smallest unit of data that can be accessed from global memory, which is exactly 32 bytes.

In the example above, all threads in a warp access consecutive memory locations both for `a`, `b`, and `c`, and for each $32 * 4 / 32 = 4$ sectors, only **ONE** instruction to a warp is needed to access the data. This is so-called **coalesced memory access**.

{{<image 
src="https://docs.google.com/drawings/d/e/2PACX-1vRXwpIJWOSYT4fXZ3ZwR8UZOXpqO0R_-AG5JLZQkm3BEZQ16KExQdAH58LkP1pvZbisQOI2-Gr0N1v_/pub?w=1006&h=371"
width="100%"
caption=`Coalesced Memory Access. There are 2N loads operations and 1N store operations in the kernel, which in all are 2N/32 load instructions and 1N/32 store instructions for warps (each warp executes 1 instruction). Since the access to memroy is coalesced, one instruction will transfer 4 sectors of data. There are no any wasted data.`
>}}

Another example of coalesced memory access is shown below:

{{< details title="Click to See Example Code" >}}
```cpp {linenos=true}
__global__ void vecAddKernelv1(const fp32_t* a, const fp32_t* b, fp32_t* c,
                               int32_t n)
{
    int gtid = threadIdx.x + blockDim.x * blockIdx.x;
    gtid = gtid % 2 == 0 ? gtid + 1 : gtid - 1;
    if (gtid < n) {
        // [DRAM] 2 load, 1 store, 3 inst
        c[gtid] = a[gtid] + b[gtid];
    }
}
```
{{< /details >}}

Crompared to the previous example, each 2 threads exchange their access positions. However, the access to memory is still coalesced.

{{<image 
src="https://docs.google.com/drawings/d/e/2PACX-1vTGbAM6z2ZZwcftUcB4E80_PUqOMCr2Y6ErnGx5DCPqVqUqFaxlDV9IbcPHjUKI1PX7v6cwcZHWH2nT/pub?w=1006&h=371"
width="100%"
caption=`Another Example of Coalesced Memory Access. 1 intruction will transfer 4 sectors of data. There are no any wasted data.`
>}}


### 2.2. Non-Coalesced Access

**Non-Coalesced Memory Access** means that some thread in a warp accesses non-consecutive memory locations so that the hardware cannot combine all these accesses into one single access. By doing so, more wasted data are transferred and the memory bandwidth is not fully utilized.

See the example code below. Originally, 32 threads in a warp would access 32 consecutive fp32 elements. However, I make the first thread in each warp access the 33th fp32 element (which should be accessed by the next warp), making an intented non-coalesced access.

{{< details title="Click to See Example Code" >}}
```cpp {linenos=true}
__global__ void vecAddKernelv1(const fp32_t* a, const fp32_t* b, fp32_t* c,
                               int32_t n)
{
    int gtid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gtid % warpSize == 0) {
        gtid = (gtid + warpSize) % (ceilDiv(n, warpSize) * warpSize);
    }
    if (gtid < n) {
        // [DRAM] 2 load, 1 store, 3 inst
        c[gtid] = a[gtid] + b[gtid];
    }
}
```
{{< /details >}}

The memory access pattern is shown in the figure below. Campare to the previous examples, you can see that despite the total number of load/store instructions is the same (2N/32 load instructions and 1N/32 store instructions), for each warp, 5 sectors of data are now being transferred per instruction. From the perspective of hardware, more data are being transferred than needed.

{{<image 
src="https://docs.google.com/drawings/d/e/2PACX-1vS26Ml2jmtIYgk4jhrnmAihGKhuGMcjnwM3aqh784REEtZVLh2_fEva6GbyaroJ9ZrF-w1QmRONlxQm/pub?w=1006&h=371"
width="100%"
caption=`Non-Coalesced Memory Access. There are 2N/32 load instructions and 1N/32 store instructions for warps. But one instruction will transfer 5 sectors of data, as shown in the first warp with 5 orange sectors.`
>}}

In Nsight Compute, you can see the performance analysis in the "Memory Workload Analysis" section. Optimization suggestions are provided for reducing wasted data transfer.

{{<image 
src="/imgs/blogs/cuda-programming-notes-01-memory-coalescing/non-coalesced-nsight-compute.png"
width="100%"
caption=`Performance analysis of non-coalesced memory access using Nsight Compute.`
>}}

## References

1. {{<href text="Programming Massively Parallel Processors: A Hands-on Approach, 4th Edition" url="https://www.elsevier.com/books/programming-massively-parallel-processors/kirk/978-0-12-811986-0">}}
1. {{<href text="【CUDA调优指南】合并访存" url="https://www.bilibili.com/video/BV1NYCtYTEFH">}}
1. {{<href text="Memory Coalescing Techniques" url="https://homepages.math.uic.edu/~jan/mcs572/memory_coalescing.pdf">}}