
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace std;

#define CHUNK_SIZE 2000
#define HASH_SIZE 1000

#define BLOCK_SIZE 512
#define ACTIVE_THREADS BLOCK_SIZE/32

__global__ void original_test(int num_level,
			     int *d_level_ptr,
			     int *d_vertices_ptr,
			     int *d_data,
			     int *d_edges,
			     int *d_tmp_output)
{
  //extern /*volatile*/ __shared__ int shared_var[];
  int global_thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  int global_warp_id = global_thread_id / 32;
  int total_threads = blockDim.x * gridDim.x;
  int total_warps = total_threads / 32;

  // int local_warp_id = threadIdx.x / 32; // local warp index
  // int total_local_warps = blockDim.x / 32;
  int lane = global_thread_id & (32 - 1); // thread index within the warp

  for (int level_id = 0; level_id < num_level; level_id++)
    {
      int start_level_ptr = d_level_ptr[level_id];
      int last_level_ptr = d_level_ptr[level_id + 1];

      for (int data_ptr = start_level_ptr + global_warp_id; data_ptr < last_level_ptr; data_ptr += total_warps)
	{
	  int id = d_data[data_ptr];
	  int first_vertex = d_vertices_ptr[id];
	  int last_vertex = d_vertices_ptr[id + 1];

	  for (int v = first_vertex + lane; v < last_vertex; v += 32)
	    {
	      int out_id = d_edges[v];
	      d_tmp_output[global_thread_id] = out_id;
	    }
	}
    }
}

__device__ int scan(int lane, int *vals, int carried_val)
{
  if (lane == 0) vals[lane] += carried_val;
  if (lane >= 1 && lane < ACTIVE_THREADS) vals[lane] += vals[lane - 1];
  if (lane >= 2 && lane < ACTIVE_THREADS) vals[lane] += vals[lane - 2];
  if (lane >= 4 && lane < ACTIVE_THREADS) vals[lane] += vals[lane - 4];
  if (lane >= 8 && lane < ACTIVE_THREADS) vals[lane] += vals[lane - 8];
  if (lane >= 16 && lane < ACTIVE_THREADS) vals[lane] += vals[lane - 16];
  return vals[31];
}

__device__ void clean_hashtable(volatile int *hash_indices, int *hash_values)
{
  for (int i = threadIdx.x; i < HASH_SIZE; i += blockDim.x)
    {
      hash_indices[i] = -1;
      hash_values[i] = -1;
    }
}

__device__ void flush_hashtable(volatile int *hash_indices, int *hash_values, int *d_hash_indices, int *d_hash_values, int base)
{
  for (int i = threadIdx.x; i < HASH_SIZE; i += blockDim.x)
    {
      d_hash_indices[base + i] = hash_indices[i];
      d_hash_values[base + i] = hash_values[i];
    }
}

__device__ void insert_hashtable(int lane, volatile int *hash_indices, int *hash_values, int id, int value)
{
	if (lane == 0)
	{
		int hash = id % HASH_SIZE;
		if (hash_indices[hash] == -1)
		{
			hash_indices[hash] = id;
			int check_id = hash_indices[hash];
			if (check_id == id)
				hash_values[hash] = value; //start_loc % CHUNK_SIZE;
		}
	}
}

__global__ void test_build_hashtable(int num_levels,
				     int *d_level_ptr,
				     int *d_vertices_ptr,
				     int *d_data,
				     int *d_edges,
				     int *d_tmp_output,
				     int *d_hash_indices,
				     int *d_hash_values,
				     int *d_chunk_level_ptr,
				     int *d_chunk_ptr)
{
  __shared__ int shared_vals[ACTIVE_THREADS];
  volatile __shared__ int chunk_id[1];
  volatile __shared__ int next_chunk_id[1];
  volatile __shared__ int hash_indices[HASH_SIZE];
  __shared__ int hash_values[HASH_SIZE];

  int global_thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  int global_warp_id = global_thread_id / 32;
  int total_threads = blockDim.x * gridDim.x;
  int total_warps = total_threads / 32;

  int local_warp_id = threadIdx.x / 32; // local warp index
  // int total_local_warps = blockDim.x / 32;
  int lane = global_thread_id & (32 - 1); // thread index within the warp
  int carried_val = 0;

  if (threadIdx.x == 0)
  {
    chunk_id[0] = 0;
    d_chunk_level_ptr[0] = 0;
    d_chunk_ptr[0] = 0;
  }

  clean_hashtable(hash_indices, hash_values);

  for (int level_id = 0; level_id < num_levels; level_id++)
    {
      int start_level_ptr = d_level_ptr[level_id];

      int last_level_ptr = d_level_ptr[level_id + 1];


      for (int data_ptr = start_level_ptr + global_warp_id; data_ptr < last_level_ptr; data_ptr += total_warps)
	{
	  int id = d_data[data_ptr];
	  int first_vertex = d_vertices_ptr[id];
	  int last_vertex = d_vertices_ptr[id + 1];

	  if (lane == 0)
	    shared_vals[local_warp_id] = last_vertex - first_vertex;

	  __syncthreads();

	  int new_carried_val;
	  if (local_warp_id == 0)
	    {
	      new_carried_val = scan(lane, shared_vals, carried_val);
	    }

	  __syncthreads();

	  int start_loc = local_warp_id == 0 ? carried_val : shared_vals[local_warp_id - 1];
	  carried_val = new_carried_val;
	  for (int v = first_vertex + lane; v < last_vertex; v += 32)
	    {
	      int out_id = d_edges[v];
	      d_tmp_output[start_loc + lane] = out_id;
	    }

	  if (shared_vals[local_warp_id] / CHUNK_SIZE == chunk_id[0])
	    {
	      insert_hashtable(lane, hash_indices, hash_values, id, start_loc % CHUNK_SIZE);
	    } 
	  else if (start_loc / CHUNK_SIZE == chunk_id[0] && lane == 0)
	    {
	      next_chunk_id[0] = chunk_id[0] + 1;
	      d_chunk_ptr[next_chunk_id[0]] = start_loc;
	    }

	  __syncthreads();

	  if (next_chunk_id[0] != chunk_id[0])
	    {
	      flush_hashtable(hash_indices, hash_values, d_hash_indices, d_hash_values, chunk_id[0] * HASH_SIZE);
	      __syncthreads();
	      clean_hashtable(hash_indices, hash_values);
	      __syncthreads();

	      if (threadIdx.x == 0)
		chunk_id[0]++;
	      
	      if (shared_vals[local_warp_id] / CHUNK_SIZE == next_chunk_id[0])
		{
		  insert_hashtable(lane, hash_indices, hash_values, id, start_loc % CHUNK_SIZE);
		}
	    }
	}
	__syncthreads();
	flush_hashtable(hash_indices, hash_values, d_hash_indices, d_hash_values, chunk_id[0] * HASH_SIZE);
	if (threadIdx.x == 0)
	{
		chunk_id[0]++;
		d_chunk_level_ptr[level_id] = chunk_id[0];
	}
	__syncthreads();
    }
}

int main(int argc, const char* argv[])
{
  cudaError_t err = cudaSuccess;
  int blocksPerGrid = 1;
  int threadsPerBlock = BLOCK_SIZE;
  int num_vertices = 0, num_edges = 0;
  int *vertices_ptr = NULL;
  int *edges = NULL;
  int *level_ptr = NULL;
  int *data = NULL;

  int *d_vertices_ptr = NULL;
  int *d_edges = NULL;
  int *d_tmp_output = NULL;
  int *d_level_ptr = NULL;
  int *d_data = NULL;
  int *d_hash_indices = NULL;
  int *d_hash_values = NULL;
  int *d_chunk_level_ptr = NULL;
  int *d_chunk_ptr = NULL;

  FILE *f = fopen(argv[1], "r");
  fscanf(f, "%d %d", &num_vertices, &num_edges);

  vertices_ptr = new int[num_vertices + 1];
  for (int i = 0; i < num_vertices + 1; i++)
    fscanf(f, "%d", &vertices_ptr[i]);

  edges = new int[num_edges];
  for (int i = 0; i < num_edges; i++)
    fscanf(f, "%d", &edges[i]);

  fclose(f);
  err = cudaMalloc((void **)&d_vertices_ptr, (num_vertices + 1) * sizeof(int));
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error code -1 %s\n", cudaGetErrorString(err));
      exit(1);
    }

  err = cudaMemcpy(d_vertices_ptr, vertices_ptr, (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error code 0 %s\n", cudaGetErrorString(err));
      exit(1);
    }
  err = cudaMalloc((void **)&d_edges, num_edges * sizeof(int));
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error code 1 %s\n", cudaGetErrorString(err));
      exit(1);
    }

  err = cudaMemcpy(d_edges, edges, num_edges * sizeof(int), cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error code 2 %s\n", cudaGetErrorString(err));
      exit(1);
    }

  f = fopen(argv[2], "r");
  int first_level = 0, last_level = 0;
  fscanf(f, "%d %d", &first_level, &last_level);
  int num_levels = last_level - first_level;
  level_ptr = new int[num_levels + 1];
  int total_size = 0;
  fscanf(f, "%d", &total_size);
  data = new int[total_size];
  int count = 0;

  for (int i = 0; i < num_levels; i++)
    {
      level_ptr[i] = count;
      int list_size = 0;
      fscanf(f, "%d", &list_size);
      for (int j = 0; j < list_size; j++)
	{
	  int tmp = 0;
	  fscanf(f, "%d", &tmp);
	  data[count + j] = tmp;
	}
      count += list_size;

    }
  level_ptr[num_levels] = total_size;
  fclose(f);

  err = cudaMalloc((void **)&d_level_ptr, (num_levels + 1) * sizeof(int));
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error code 3 %s\n", cudaGetErrorString(err));
      exit(1);
    }

  err = cudaMemcpy(d_level_ptr, level_ptr, (num_levels + 1) * sizeof(int), cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error code 4 %s\n", cudaGetErrorString(err));
      exit(1);
    }
  err = cudaMalloc((void **)&d_data, total_size * sizeof(int));
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error code 5 %s\n", cudaGetErrorString(err));
      exit(1);
    }

  err = cudaMemcpy(d_data, data, total_size * sizeof(int), cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error code 6 %s\n", cudaGetErrorString(err));
      exit(1);
    }

  err = cudaMalloc((void **)&d_tmp_output, (total_size * 3) * sizeof(int));
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error code 1 %s\n", cudaGetErrorString(err));
      exit(1);
    }

  err = cudaMalloc((void **)&d_hash_indices, 2 * total_size * sizeof(int));
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error code 7 %s\n", cudaGetErrorString(err));
      exit(1);
    }

  err = cudaMalloc((void **)&d_hash_values, 2 * total_size * sizeof(int));
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error code 8 %s\n", cudaGetErrorString(err));
      exit(1);
    }

  err = cudaMalloc((void **)&d_chunk_level_ptr, (num_levels + 1) * sizeof(int));
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error code 9 %s\n", cudaGetErrorString(err));
      exit(1);
    }

  err = cudaMalloc((void **)&d_chunk_ptr, total_size * sizeof(int));
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error code 10 %s\n", cudaGetErrorString(err));
      exit(1);
    }


  printf("Start\n");
//   original_test<<<blocksPerGrid, threadsPerBlock>>>(num_levels,
// 						    d_level_ptr,
// 						    d_vertices_ptr,
// 						    d_data,
// 						    d_edges,
// 						    d_tmp_output);

  test_build_hashtable<<<blocksPerGrid, threadsPerBlock>>>( num_levels,
							    d_level_ptr,
							    d_vertices_ptr,
							    d_data,
							    d_edges,
							    d_tmp_output,
							    d_hash_indices,
							    d_hash_values,
							    d_chunk_level_ptr,
							    d_chunk_ptr);

  err =  cudaDeviceSynchronize();
  if (err != cudaSuccess)
    {
      fprintf(stderr, "error code 11 %s\n", cudaGetErrorString(err));
      exit(1);
    }

  if (vertices_ptr != NULL) delete[] vertices_ptr;
  if (edges != NULL) delete[] edges;
  if (level_ptr != NULL) delete[] level_ptr;
  if (data != NULL) delete[] data;

  if (d_vertices_ptr != NULL) cudaFree(d_vertices_ptr);
  if (d_edges != NULL) cudaFree(d_edges);
  if (d_tmp_output != NULL) cudaFree(d_tmp_output);
  if (d_level_ptr != NULL) cudaFree(d_level_ptr);
  if (d_hash_indices != NULL) cudaFree(d_hash_indices);
  if (d_hash_values != NULL) cudaFree(d_hash_values);
  if (d_chunk_level_ptr != NULL) cudaFree(d_chunk_level_ptr);
  if (d_chunk_ptr != NULL) cudaFree(d_chunk_ptr);

  return 0;
}

