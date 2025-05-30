#include <mpi.h>
#include <math.h>

#include "algorithms.h"

void merge_2_way(tuwtype_t *to_merge, int first_size, int second_size){
  int mergecount = first_size + second_size;
  tuwtype_t *temp = NULL;
  // temporary buffer for merging
  temp = (tuwtype_t *)malloc(mergecount * sizeof(tuwtype_t));

  // pointers to the start of each block
  tuwtype_t *first = to_merge;
  //tuwtype_t *second = &to_merge[blocksize * s_k];
  tuwtype_t *second = to_merge + first_size;

  // indices for merging
  int i = 0; int j = 0; int k = 0;
  // loop until one block is empty
  while(i < first_size && j < second_size){
      // check for smaller item, add it to the temporary merged array and update indices
      if(first[i] <= second[j]){
        temp[k] = first[i];
        k++; i++;
      }else{
        temp[k] = second[j];
        k++; j++;
      }
  }
// after 1 block is finished, add remaining elements to temp
while (i < first_size){
  temp[k] = first[i];
  k++; i++;
}
while (j < second_size){
  temp[k] = second[j];
  k++; j++;
}
// copy back merged array
memcpy(to_merge, temp, mergecount * sizeof(tuwtype_t));
free(temp);
}

int is_power_of_two(int n) {
  return n > 0 && (n & (n - 1)) == 0;
}

int HPC_AllgatherMergeBruck(const void *sendbuf, int sendcount,
                            MPI_Datatype sendtype, void *recvbuf, int recvcount,
                            MPI_Datatype recvtype, MPI_Comm comm) 
{
  int size;
  MPI_Comm_size(comm, &size);
  // calculate total elements of recvbuf
  if(size == 1){
    memcpy(recvbuf, sendbuf, recvcount * sizeof(tuwtype_t));
    return MPI_SUCCESS;
  }
  int buf_size = size * recvcount;

  // get rank of process
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // array for merged elements
  tuwtype_t *merged = NULL;
  tuwtype_t *unmerged = NULL;
  merged = (tuwtype_t *)malloc(buf_size * sizeof(tuwtype_t));
  unmerged = (tuwtype_t *)malloc(buf_size * sizeof(tuwtype_t));

  //here initialize merged with the first block(treat it as B in the description)
  memcpy(merged, sendbuf, recvcount * sizeof(tuwtype_t));
  // keep track of the unmerged blocks that are sent for the last round
  memcpy(unmerged, sendbuf, recvcount * sizeof(tuwtype_t));

  int q = ceil(log2(size));
  int s_k = 0;
  int to = 0;
  int from = 0;
  int tag = 0;
  int s_kp = 0;
  // if not a power of 2 calculate number of blocks to send in last round
  int num_last_rnd = size - exp2(q-1);
  // idea: save the first #num_last_rnd blocks in an extra array for the last round and merge them
  tuwtype_t *last_round = NULL;
  last_round = (tuwtype_t *)malloc(num_last_rnd * sendcount * sizeof(tuwtype_t));
  // if not a power of 2 then initialize last_round with the first block
  memcpy(last_round, sendbuf, recvcount * sizeof(tuwtype_t));
  // int to keep track of how many blocks for the last round are already saved
  int num_last_rnd_saved = 1;
  int num_last_rnd_new = 1;
  // to keep track of remaining blocks to
  int difference = 0;
  for(int k = 0; k < q; k++){
    difference = num_last_rnd - num_last_rnd_saved;
    s_k = exp2(k);
    // s_k+1
    if(k < (q-1)){
      s_kp = s_k*2;
    }else{
      s_kp = size;
    }
    // calculate rank of process to send the data to
    to = (rank - s_k + size) % size;
    // calculate rank of process to receive the data from
    from = (rank + s_k) % size;
    // use mpi send and receive call to send and receive the correct data block
    //check for last round
    if(!is_power_of_two(size) && k == (q-1)){
      MPI_Sendrecv(last_round, (s_kp-s_k) * sendcount, MPI_INT, to, tag, &merged[s_k * sendcount], (s_kp-s_k) * sendcount, MPI_INT, from, tag, comm, MPI_STATUS_IGNORE);
    }else{
      MPI_Sendrecv(merged, (s_kp-s_k) * sendcount, MPI_INT, to, tag, &merged[s_k * sendcount], (s_kp-s_k) * sendcount, MPI_INT, from, tag, comm, MPI_STATUS_IGNORE);
      MPI_Sendrecv(unmerged, (s_kp-s_k) * sendcount, MPI_INT, to, tag, &unmerged[s_k * sendcount], (s_kp-s_k) * sendcount, MPI_INT, from, tag, comm, MPI_STATUS_IGNORE);
    }
    
    // if number of remaining blocks for last round not already reached, save the correct number of just received blocks
    if (difference > 0 ){
      if(difference >= s_k){
        memcpy(&last_round[num_last_rnd_saved * sendcount], &merged[s_k * sendcount], s_k*sendcount*sizeof(tuwtype_t));
        num_last_rnd_new = num_last_rnd_saved + s_k;
      }else{ // blocks for last round, coming from the unmerged array
        memcpy(&last_round[num_last_rnd_saved * sendcount], &unmerged[s_k * sendcount], difference*sendcount*sizeof(tuwtype_t));
        num_last_rnd_new = num_last_rnd_saved + difference;
      }
      
    }
    //merge_2_way(merged, s_kp-s_k, sendcount);
    merge_2_way(merged, s_k*sendcount, (s_kp-s_k)*sendcount);

    // merge blocks for last round
    if(difference > 0){
      merge_2_way(last_round, num_last_rnd_saved*sendcount, (num_last_rnd_new-num_last_rnd_saved)*sendcount);
    }
    // update number of saved blocks
    num_last_rnd_saved = num_last_rnd_new;
  }
  memcpy(recvbuf, merged, buf_size * sizeof(tuwtype_t));
  free(merged);
  free(last_round);  
  free(unmerged);                     
  return MPI_SUCCESS;
}