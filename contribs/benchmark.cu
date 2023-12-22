
// sur hfgpu
// nvcc -arch=compute_80 -code=sm_80 --compiler-options -fopenmp benchmark.cu -o benchmark_hfgpu
// execavec memoire unifiee
// OMP_NUM_THREADS=64 ccc_mprun -n1 -c128 -phfgpu -T3600 ./benchmark_hfgpu <<< "0 1 1037"
// avec memoire device et copie host/device
// OMP_NUM_THREADS=64 ccc_mprun -n1 -c128 -phfgpu -T3600 ./benchmark_hfgpu <<< "0 0 1037"

// sur HE
// nvcc -arch=compute_80 -code=sm_80 --compiler-options -fopenmp benchmark.cu -o benchmark_he
// nvcc -arch=compute_90 -code=sm_90 --compiler-options -fopenmp benchmark.cu -o benchmark_he
// execavec memoire unifiee
// OMP_NUM_THREADS=64 ./benchmark_he <<< "0 1 1037"
// avec memoire device et copie host/device
// OMP_NUM_THREADS=64 ./benchmark_he <<< "0 0 1037"

#include <iostream>
#include <chrono>
#include <omp.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define N (1024*1024)
#define M (10000000)

void cpu_init( double * __restrict__ data )
{
# pragma omp parallel
  {
#   pragma omp single
    {
      std::cout << "using "<<omp_get_num_threads()<<" CPU threads"<<std::endl;
    }
#   pragma omp for schedule(static)
    for(int i = 0; i < N; i++)
    {
      data[i] = i * 1.0 / N;
    }
  }
}

void cpu_compute( double * __restrict__ data )
{   
# pragma omp parallel for schedule(static)
  for(int i = 0; i < N; i++)
  {
    for(int j = 0; j < M; j++)
    {
       data[i] = data[i] * data[i] - 0.25;
    }
  }
}

__global__ void gpu_compute( double * __restrict__ data )
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  for(int j = 0; j < M; j++)
  {
    data[i] = data[i] * data[i] - 0.25;
  }
}

int main()
{
  double *h_data = nullptr;
  double *d_data = nullptr;

  int run_host=0, uvm=0, idx=0;
  std::cin >> run_host >> uvm >> idx; 
  std::cout << "run_host="<<run_host<<", uvm="<<uvm<<" , idx="<<idx<<std::endl;

  if( uvm )
  {
    cudaMallocManaged( & h_data , N * sizeof(double) );
    d_data = h_data;
  }
  else
  {
    h_data = new double[N];
  }

  cpu_init( h_data );

  if( ! uvm )
  {
    cudaMalloc( & d_data, N * sizeof(double));
    cudaMemcpy( d_data, h_data, N * sizeof(double), cudaMemcpyHostToDevice );
  }

  const auto T0 = std::chrono::high_resolution_clock::now();

  if(run_host) cpu_compute( h_data );
  const double vhost = h_data[idx];

  const auto T1 = std::chrono::high_resolution_clock::now();

  gpu_compute<<<N/256, 256>>>(d_data);   
  const auto T2 = std::chrono::high_resolution_clock::now();

  if( ! uvm ) cudaMemcpy( h_data, d_data, N * sizeof(double), cudaMemcpyDeviceToHost );
  cudaDeviceSynchronize();
  const double vcuda = h_data[idx];
  const auto T3 = std::chrono::high_resolution_clock::now();

  std::cout << "result["<<idx<<"] = "<< vhost<<" / "<<vcuda<<std::endl;
  if(run_host) std::cout << "host time = "<< (T1-T0).count() / 1000000.0 << std::endl;
  std::cout << "cuda time = "<< (T2-T1).count() / 1000000.0 << " + "<< (T3-T2).count() / 1000000.0 << " = "<< (T3-T1).count() / 1000000.0 <<std::endl;
  if(run_host) std::cout << "ratio = "<< (T1-T0).count() * 1.0 / (T3-T1).count()  << std::endl;

  if( ! uvm ) cudaFree(d_data); 
  
  return 0;
}

