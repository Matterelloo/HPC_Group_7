#include <mpi.h>
#include <vector>
#include <queue>
#include <cstring>
#include "algorithms.h"

// struct of block element with int value, index of the block it is in and index in the block
struct BlockElement {
  tuwtype_t val;
  int block_idx;
  int idx_in_block;

  BlockElement(tuwtype_t v, int b, int i) : val(v), block_idx(b), idx_in_block(i) {}
};

struct CompareNodes {
  bool operator()(const BlockElement& a, const BlockElement& b) const {
      return a.val > b.val; // Min-heap
  }
};

int HPC_AllgatherMergeBase(const void *sendbuf, int sendcount,
                           MPI_Datatype sendtype, void *recvbuf, int recvcount,
                           MPI_Datatype recvtype, MPI_Comm comm) {
  
  // call Allgather to collect each block at each process
  MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
  tuwtype_t *data = static_cast<tuwtype_t *>(recvbuf);

  // get size of communicator
  int size;
  MPI_Comm_size(comm, &size);
  // calculate total elements of recvbuf
  int buf_size = size * recvcount;

    // priority queue
    std::priority_queue<BlockElement, std::vector<BlockElement>, CompareNodes> minHeap;
    std::vector<tuwtype_t> merged;
    merged.reserve(buf_size);

    // push the first element from each block at the beginning
    for (int i = 0; i < size; i++) {
        int first = i * recvcount;
        if (first < buf_size) {
            // construct BlockElement and push it into the heap
            minHeap.emplace(data[first], i, 0);
        }
    }

    // p-way merge
    while (!minHeap.empty()) {
        // smallest element
        BlockElement smallest_element = minHeap.top();
        // remove from heap
        minHeap.pop();
        // add value to merged
        merged.push_back(smallest_element.val);

        int next_idx = smallest_element.idx_in_block + 1;
        int block_start = smallest_element.block_idx * recvcount;
        int next_global_idx = block_start + next_idx;

        // add next value (if any) of the block to the heap
        if (next_idx < recvcount && next_global_idx < buf_size) {
            minHeap.emplace(data[next_global_idx], smallest_element.block_idx, next_idx);
        }
    }

    // copy back the sorted data into recvbuf
    std::memcpy(recvbuf, merged.data(), buf_size * sizeof(tuwtype_t));
  return MPI_SUCCESS;
}
