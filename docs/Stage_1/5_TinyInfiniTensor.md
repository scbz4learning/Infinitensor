# AI Complier

## HW 1 - Allocator

### Question Understanding
```c++
#pragma once
#include "core/runtime.h"
#include "core/tensor.h"
#ifdef BUILD_TEST
#include "gtest/gtest.h"
#endif
#include <cstddef>
#include <map>
#include <unordered_set>

namespace infini {
  class Allocator
  {
  private:
    Runtime runtime;

    size_t used;

    size_t peak;

    size_t alignment;

    // pointer to the memory actually allocated
    void *ptr;

    // =================================== 作业 ===================================
    // TODO：可能需要设计一个数据结构来存储free block，以便于管理和合并
    // HINT: 可以使用一个 map 来存储 free block，key 为 block 的起始/结尾地址，value 为 block 的大小
    // =================================== 作业 ===================================
    std::map<size_t, size_t> free_blocks;

  public:
    Allocator(Runtime runtime);

    virtual ~Allocator();

    // function: simulate memory allocation
    // arguments：
    //     size: size of memory block to be allocated
    // return: head address offset of the allocated memory block
    size_t alloc(size_t size);

    // function: simulate memory free
    // arguments:
    //     addr: head address offset of memory block to be free
    //     size: size of memory block to be freed
    void free(size_t addr, size_t size);

    // function: perform actual memory allocation
    // return: pointer to the head address of the allocated memory
    void *getPtr();

    void info();

  private:
    // function: memory alignment, rouned up
    // return: size of the aligned memory block
    size_t getAlignedSize(size_t size);
  };
}
```

The allocator is for uniformly manage the ram used in a graph. However, it is just a table while the tensor will still need to allocate the mem from OS.  

For the prototype, we need a map. 

```c++
std::map<size_t, size_t> free_blocks;
```

The initiation is to allocate the mem statically like early OS, where the allocator needs to create new nodes and merge nodes of free spaces. The codes should be similar to 

```c++
for (; it != free_blocks.end(); it++) {
    if (it->second >= size) {
        offset = it->first;
        if (it->second > size) {
            free_blocks.emplace(offset+size, it->second-size);
        }
        // Without this line, it will be an undefined behaviour
        // to test if `it == free_blocks.end()`
        free_blocks.erase(it);
        it = free_blocks.begin();
        break;
    }
}
```

However, there is big problem - we do not know the initial value. Should we record the allocated mem? It is better to know the perpose. The real project of this tiny teaching model has the [allocator](https://github.com/InfiniTensor/InfiniTensor/blob/master/src/core/lazy_allocator.cc#L61) as well: 

```c++
size_t LazyAllocator::alloc(size_t size) {
    // pad the size to the multiple of alignment
    size = this->getAlignedSize(size);
    auto it = this->freeBlocks.lower_bound(freeBlockInfo{(size_t)0, size});

    size_t retAddr = this->peak;
    if (it != this->freeBlocks.end()) {
        // found an alvailable free memory block for allocation
        ...
    } else {
        // the allocated memory space is not sufficient for reallocation, it
        ...
    }

    return retAddr;
}
```

So the reason is clear - the allocator is to maintain the allocate but freed mem, to reduce the fregments, maintain the peak mem usage and help optimise the graph nodes.

Therefore,  
1. if there is enough space, allocate the mem in the table with merging  
2. else reassign peak / used and add a node in the map

### Implementation
#### First-fit with $O(\log n)$ Search
The first intitiation is to use a `std::map` to store all the pairs of `startAddr` and `size` with . For allocation, just do linear scan from the beginning and same for freeing. However, this cost $O(\log n)$ time for searching, erasing and deleting. Plus, the first-fit strategy can produce more fregments than best fit, which is more commonly used in practice.  

#### Best-fit with $O(\log n)$ Search  
It is then very straight forward to implement a balanced tree in the order according to size, and a hashmap with doubly linked list from addresses to values for quicker searching. The prototype should be like:  
```c++
struct BlockInfo {
    size_t addr;
    size_t size;

    bool operator<(const BlockInfo &other) const {
        return (size != other.size) ? (size < other.size) : (addr < other.addr);
    }
};

// Balanced tree for free memory blocks, sorted by size and address
std::set<BlockInfo> freeBlocks;

// Key: Starting address of the free memory block
// Value: Size of the block
std::unordered_map<size_t, size_t> blockStartToSize;
```

However, there are 2 problems:  
1. when the peak mem must be expanded, we can only do linear search for the address - Add a `tailBlock` pointer;  
2. When freeing the blocks, we cannot find the adjacent blocks. - 
  
The last one is a big problem. We must use `size` and `addr` to find the block, causing `Oo(\log n)` time, which is still good, as eraseing and inserting new blocks will definately take `O(\log n)` as well.

Another thinking is to insert all the blocks (both used or free) with a dirty tag. However, we only extend the peak when we exceed it - i.e. commonly there must be a lot more used blocks than free blocks, resulting in a much worse time consuming.

#### The Official Implementation 
One thing reminds me.

> It is so straightforward only to record the `startAddr` and `size`. But why the assignment instruction also mentioned the `endAddr`?  

The reason is, the official implementation gives a balanced tree for blocks in order of size, and 2 hash maps for both start and end addresses! The benefit is:  **we can know where is the adjacent blocks if it exist!** If there is a block before the newly freed one, its `endAddr` must be my `startAddr`, so I can find it from the `endAddrToSize` hash map and same for the block next. In this way, the searching cost $O(1)$ only with erasing and inserting still $O(\log n)$ unavoidably.

#### More Researchs
However, it must a fragile structure to maintain 3 datastructes. What is the implementation in pytorch?

> SHOCKINGLY, no merging at all!

Pytorch does not merging free spaces at all. Even if a new block is sufficient to save in two adjacent small blocks, it still requires a new space. Of course this is causing $O(1)$ time only, but that is why we need such a big mem GPU...

TBH, it must be better to use some time to merge. Learning rom OS file system, the time consumption for searching and merging index blocks rather than data blocks will not consume too much time.

## HW 2