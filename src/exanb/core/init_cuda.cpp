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
    using DeviceLimitsMap = std::map<std::string,std::string>;
  
    ADD_SLOT( MPI_Comm , mpi         , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( bool     , single_gpu  , INPUT , true ); // how to partition GPUs inside groups of contiguous MPI ranks
    ADD_SLOT( long     , rotate_gpu  , INPUT , 0 );    // shift gpu index : gpu device index assigned to an MPI process p, when single_gpu is active, is ( p + rotate_gpu ) % Ngpus. Ngpus being the number of GPUs per numa node.
    ADD_SLOT( long     , smem_bksize , INPUT , OPTIONAL );
    ADD_SLOT( DeviceLimitsMap , device_limits  , INPUT , OPTIONAL );
    ADD_SLOT( bool     , enable_cuda , INPUT_OUTPUT , true );

  public:

    inline void execute () override final
    {
      MPI_Comm comm = *mpi;
      int nprocs = 1;
      int rank = 0;
      MPI_Comm_size(comm,&nprocs);
      MPI_Comm_rank(comm,&rank);

      lout << "=========== Cuda ================"<<std::endl;

#     ifdef XSTAMP_CUDA_VERSION

      std::shared_ptr<onika::cuda::CudaContext> cuda_ctx = nullptr;

      int n_gpus = 0;
      if( *enable_cuda ) cudaGetDeviceCount(&n_gpus);
      if( n_gpus <= 0 )
      {
        lout <<"No GPU found"<<std::endl;
//        cuda_ctx->m_devices.clear();
//        cuda_ctx->m_threadStream.clear();
        onika::memory::GenericHostAllocator::set_cuda_enabled( false );
      }
      else
      {
        cuda_ctx = std::make_shared<onika::cuda::CudaContext>();
      
        int gpu_first_device = 0;
        if( *single_gpu )
        {
          gpu_first_device = ( rank + (*rotate_gpu) ) % n_gpus;
          cuda_ctx->m_devices.resize(1);
        }
        else
        {
          cuda_ctx->m_devices.resize(n_gpus);
        }
      
        int ndev = cuda_ctx->m_devices.size();
        for(int d=0;d<ndev;d++) cuda_ctx->m_devices[d].device_id = gpu_first_device + d;
        
        const int max_threads = omp_get_max_threads();
        if( max_threads < ndev )
        {
          fatal_error()<<"Unsupported configuration: number of threads ("<<max_threads<<") less than number of GPUs ("<<ndev<<")"<<std::endl;
        }
        ldbg << "support for a maximum of "<<max_threads<<" threads accessing "<<ndev<<" GPUs"<<std::endl;
        //cuda_ctx->m_threadStream.resize( ndev , 0 );
        assert( ndev > 0 );
        
        checkCudaErrors( cudaSetDevice( cuda_ctx->m_devices[0].device_id ) );
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
        
        if( device_limits.has_value() )
        {
          auto m = *device_limits;
          if( m.find("query_all") != m.end() )
          {
            m.clear();
            m["cudaLimitStackSize"] = "-1";
            m["cudaLimitPrintfFifoSize"] = "-1";
            m["cudaLimitMallocHeapSize"] = "-1";
            m["cudaLimitDevRuntimeSyncDepth"] = "-1";
            m["cudaLimitDevRuntimePendingLaunchCount"] = "-1";
            m["cudaLimitMaxL2FetchGranularity"] = "-1";
            m["cudaLimitPersistingL2CacheSize"] = "-1";
          }
          for( const auto& dl : m )
          {
            cudaLimit limit;
                 if( dl.first == "cudaLimitStackSize"                    ) limit = cudaLimitStackSize;
            else if( dl.first == "cudaLimitPrintfFifoSize"               ) limit = cudaLimitPrintfFifoSize;
            else if( dl.first == "cudaLimitMallocHeapSize"               ) limit = cudaLimitMallocHeapSize;
            else if( dl.first == "cudaLimitDevRuntimeSyncDepth"          ) limit = cudaLimitDevRuntimeSyncDepth;
            else if( dl.first == "cudaLimitDevRuntimePendingLaunchCount" ) limit = cudaLimitDevRuntimePendingLaunchCount;
            else if( dl.first == "cudaLimitMaxL2FetchGranularity"        ) limit = cudaLimitMaxL2FetchGranularity;
            else if( dl.first == "cudaLimitPersistingL2CacheSize"        ) limit = cudaLimitPersistingL2CacheSize;
            else
            {
              fatal_error() << "Cuda unknown limit '"<<dl.first<<"'"<<std::endl;
            }
            long in_value = std::stol( dl.second );
            
            if( in_value >= 0 )
            {
              checkCudaErrors( cudaDeviceSetLimit( limit , in_value ) ); 
            }
            else
            {
              size_t value = 0;
              checkCudaErrors( cudaDeviceGetLimit ( &value, limit ) );
              lout << dl.first << " = " << value << std::endl;
            }
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
        onika::memory::GenericHostAllocator::set_cuda_enabled( true );

        lout <<"GPUs : "<<ndev<< std::endl;
        lout <<"Type : "<<device_name << std::endl;
        lout <<"SMs  : "<<multiProcessorCount<<"x"<<warpSize<<" threads @ "<< std::defaultfloat<< clock_rate/1000000.0<<" Ghz" << std::endl;
        lout <<"Mem  : "<< memory_bytes_string(totalGlobalMem/ndev) <<" (shared="<<memory_bytes_string(sharedMemPerBlock,"%g%s")<<" L2="<<memory_bytes_string(l2_cache,"%g%s")<<")" <<std::endl;
      }
      
      set_global_cuda_ctx( cuda_ctx );
            
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


