#include <mpi.h>
#include <math.h>
#include "algorithms.h"

void merge(tuwtype_t *A, int sizeA, tuwtype_t *B, int sizeB, tuwtype_t* result) {
  int i = 0, j = 0, k = 0;
  while (i < sizeA && j < sizeB) {
      if (A[i] <= B[j]) result[k++] = A[i++];
      else result[k++] = B[j++];
  }
  while (i < sizeA) result[k++] = A[i++];
  while (j < sizeB) result[k++] = B[j++];
}

int HPC_AllgatherMergeCirculant(const void *sendbuf, int sendcount,
                                MPI_Datatype sendtype, void *recvbuf,
                                int recvcount, MPI_Datatype recvtype,
                                MPI_Comm comm)
{
  int size;
  MPI_Comm_size(comm, &size);
  // return sendbuf if size = 1
  if(size == 1){
    memcpy(recvbuf, sendbuf, sendcount * sizeof(sendtype));
    return MPI_SUCCESS;
  }
  // calculate total elements of recvbuf
  int buf_size = size * recvcount; 

  // get rank of process
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  int q = ceil(log2(size));
  // generate s_k array
  int* s_k = (int *)malloc((q + 1) * sizeof(int));
  s_k[q] = size;
  for (int k = q - 1; k >= 0; k--){
      s_k[k] = (s_k[k + 1] + 1) / 2;
  }
  // allocate buffers for algorithm
  tuwtype_t *W = NULL;
  tuwtype_t *W_p = NULL;
  tuwtype_t *T = NULL;
  tuwtype_t *tmp = NULL;
  W = (tuwtype_t *)malloc(buf_size * sizeof(tuwtype_t));
  W_p = (tuwtype_t *)malloc(buf_size * sizeof(tuwtype_t));
  T = (tuwtype_t *)malloc(buf_size * sizeof(tuwtype_t));
  tmp = (tuwtype_t *)malloc(buf_size * sizeof(tuwtype_t));

  // to keep track of the size of W
  int W_size = recvcount;

  int eps = 0;
  int to = 0;
  int from = 0;
  int tag = 0;

  for (int k = 0; k < q; k++) {
    eps = s_k[k + 1] & 1; 
    to = (rank - s_k[k] + eps + size) % size;
    from = (rank + s_k[k] - eps) % size;

    if(eps == 1){
      // sizes??
      MPI_Sendrecv(W, W_size, MPI_INT, to, tag, T, W_size, MPI_INT, from, tag, comm, MPI_STATUS_IGNORE);
      merge(W, W_size, T, W_size, tmp);
      // adjust pointer 
      tuwtype_t *tmp_ptr = W;
      W = tmp;
      tmp = tmp_ptr;
      // update size of W
      W_size = W_size * 2;
    }else{
      if(k == 0){
        // send and receive first block
        MPI_Sendrecv(sendbuf, sendcount, MPI_INT, to, tag, W, recvcount, MPI_INT, from, tag, comm, MPI_STATUS_IGNORE);
      }else{
        // TODO merge W and sendbuf and safe in W_p
        merge((tuwtype_t *)sendbuf, sendcount, W, W_size, W_p);
        MPI_Sendrecv(W_p, W_size + sendcount, MPI_INT, to, tag, T, W_size + sendcount, MPI_INT, from, tag, comm, MPI_STATUS_IGNORE);
        merge(W, W_size, T, W_size + sendcount, tmp);
        int *tmp_ptr2 = W;
        W = tmp;
        tmp = tmp_ptr2;
        W_size = 2 * W_size + sendcount;
      }
    }
  }
  merge((tuwtype_t *)sendbuf, sendcount, W, W_size, tmp);
  memcpy(recvbuf, tmp, buf_size * sizeof(tuwtype_t));

  free(s_k);
  free(W); 
  free(T);
  free(tmp);
  return MPI_SUCCESS;
}
