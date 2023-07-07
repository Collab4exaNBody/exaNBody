#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/log.h>
#include <exanb/core/string_utils.h>
#include <exanb/core/value_streamer.h>

#ifdef XSTAMP_CUDA_VERSION
#include <onika/cuda/cuda_context.h>
#include <onika/cuda/cuda_error.h>
#include <cuda_runtime.h>
#endif

#include <onika/memory/allocator.h>

#include <omp.h>
#include <mpi.h>

#include <ios>

namespace exanb
{

  class InitCuda : public OperatorNode
  {

#ifdef XSTAMP_CUDA_VERSION
    ADD_SLOT( onika::cuda::CudaContext , cuda_ctx , OUTPUT );
#endif

    ADD_SLOT( MPI_Comm , mpi         , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( bool     , single_gpu  , INPUT , true ); // how to partition GPUs inside groups of contiguous MPI ranks
    ADD_SLOT( long     , smem_bksize , INPUT , OPTIONAL );
    ADD_SLOT( bool     , enable_cuda , INPUT , true );

  public:

    inline bool is_sink() const override final { return true; } // not a suppressable operator

    inline void execute () override final
    {
      MPI_Comm comm = *mpi;
      int nprocs = 1;
      int rank = 0;
      MPI_Comm_size(comm,&nprocs);
      MPI_Comm_rank(comm,&rank);

      lout << "=========== Cuda ================"<<std::endl;
#     ifdef XSTAMP_CUDA_VERSION
      int n_gpus = 0;
      if( *enable_cuda ) cudaGetDeviceCount(&n_gpus);
      if( n_gpus <= 0 )
      {
        lout <<"No GPU found"<<std::endl;
        cuda_ctx->m_devices.clear();
        cuda_ctx->m_threadStream.clear();
        onika::memory::GenericHostAllocator::set_cuda_enabled( false );
      }
      else
      {
        int gpu_first_device = 0;
        if( *single_gpu )
        {
          gpu_first_device = rank % n_gpus;
          cuda_ctx->m_devices.resize(1);
        }
        else
        {
          cuda_ctx->m_devices.resize(n_gpus);
        }
      
        int ndev = cuda_ctx->m_devices.size();
        for(int d=0;d<ndev;d++) cuda_ctx->m_devices[d].device_id = gpu_first_device + d;
        
        int max_threads = omp_get_max_threads();
        if( max_threads < ndev )
        {
          lerr<<"Unsupported configuration: number of threads ("<<max_threads<<") less than number of GPUs ("<<ndev<<")"<<std::endl;
          std::abort();
        }
        ldbg << "support for a maximum of "<<max_threads<<" threads accessing "<<ndev<<" GPUs"<<std::endl;
        cuda_ctx->m_threadStream.resize( ndev , 0 );
        assert( ndev > 0 );
        checkCudaErrors( cudaSetDevice( cuda_ctx->m_devices[0].device_id ) );

#       pragma omp parallel //num_threads(ndev)
        {
          const size_t tid = omp_get_thread_num();
          unsigned int gpu_index = tid % ndev;
          checkCudaErrors( cudaSetDevice( cuda_ctx->m_devices[gpu_index].device_id ) );
          if( smem_bksize.has_value() )
          {
            switch( *smem_bksize )
            {
              case 4 : checkCudaErrors(  cudaDeviceSetSharedMemConfig( cudaSharedMemBankSizeFourByte ) ); break;
              case 8 : checkCudaErrors(  cudaDeviceSetSharedMemConfig( cudaSharedMemBankSizeEightByte ) ); break;
              default:
                lerr<<"Unsupported shared memory bank size "<<*smem_bksize<<", using default\n";
                checkCudaErrors(  cudaDeviceSetSharedMemConfig( cudaSharedMemBankSizeDefault ) );
                break;
            }
          }
          if( tid < size_t(ndev) )
          {
            assert( tid < cuda_ctx->m_threadStream.size() );
            checkCudaErrors( cudaStreamCreateWithFlags( & cuda_ctx->m_threadStream[tid], cudaStreamNonBlocking ) );
          }
        }

        int n_support_vmm = 0;
        long long totalGlobalMem = 0;
        int warpSize = 0;
        int multiProcessorCount = 0;
        int sharedMemPerBlock = 0;
        int clock_rate = 0;
        int l2_cache = 0;
        std::string device_name;

        for(int i=0;i<ndev;i++)
        {
          checkCudaErrors( cudaGetDeviceProperties( & cuda_ctx->m_devices[i].m_deviceProp , i + gpu_first_device ) );
          if( i==0 ) { device_name = cuda_ctx->m_devices[i].m_deviceProp.name; }
          else if( device_name != cuda_ctx->m_devices[i].m_deviceProp.name ) { lerr<<"WARNING: Mixed GPU devices"<<std::endl; }
          bool mm = cuda_ctx->m_devices[i].m_deviceProp.managedMemory;
          bool cma = cuda_ctx->m_devices[i].m_deviceProp.concurrentManagedAccess;
          if( mm && cma ) { ++ n_support_vmm; }
          totalGlobalMem += cuda_ctx->m_devices[i].m_deviceProp.totalGlobalMem ;
          warpSize = cuda_ctx->m_devices[i].m_deviceProp.warpSize;
          multiProcessorCount = cuda_ctx->m_devices[i].m_deviceProp.multiProcessorCount;
          sharedMemPerBlock = cuda_ctx->m_devices[i].m_deviceProp.sharedMemPerBlock;
          clock_rate = cuda_ctx->m_devices[i].m_deviceProp.clockRate;
          l2_cache = cuda_ctx->m_devices[i].m_deviceProp.persistingL2CacheMaxSize;
        }

        long long tmp[3];
        ValueStreamer<long long>(tmp) << ndev << n_support_vmm << totalGlobalMem;
        MPI_Allreduce(MPI_IN_PLACE,tmp,3,MPI_LONG_LONG,MPI_SUM,*mpi);
        ValueStreamer<long long>(tmp) >> ndev >> n_support_vmm >> totalGlobalMem;

	if( n_support_vmm != ndev )
	{
	  lerr<<"GPUs don't support unified memory, cannot continue"<<std::endl;
	  std::abort();
	}

        lout <<"GPUs : "<<ndev<< std::endl;
        lout <<"Type : "<<device_name << std::endl;
        lout <<"SMs  : "<<multiProcessorCount<<"x"<<warpSize<<" threads @ "<< std::defaultfloat<< clock_rate/1000000.0<<" Ghz" << std::endl;
        lout <<"Mem  : "<< memory_bytes_string(totalGlobalMem/ndev) <<" (shared="<<memory_bytes_string(sharedMemPerBlock,"%g%s")<<" L2="<<memory_bytes_string(l2_cache,"%g%s")<<")" <<std::endl;
      }
      
      // FIXME: this works only because all operator nodes share the same global parallel task queue
      // this is not very clean, should be rethinked
      ptask_queue().set_cuda_ctx( & (*cuda_ctx) );
      
#     else
      lout <<"Cuda disabled"<<std::endl;
#     endif
      lout << "================================="<<std::endl<<std::endl;
    }
  };
  
  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "init_cuda", make_compatible_operator< InitCuda > );
  }

}


