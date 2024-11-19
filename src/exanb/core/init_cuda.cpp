/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/log.h>
#include <onika/string_utils.h>
#include <onika/value_streamer.h>

#ifdef ONIKA_CUDA_VERSION
#include <onika/cuda/cuda_context.h>
#include <onika/cuda/cuda_error.h>
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
    ADD_SLOT( bool     , mpi_even_only  , INPUT , false , DocString{"if set to true, only even MPI will be assigned a GPU device. device id assigned is then (rank/2)%ndev instead of rank%ndev (ndev being number of GPUs per node)"} );
    ADD_SLOT( long     , rotate_gpu  , INPUT , 0 , DocString{"shift gpu index : gpu device index assigned to an MPI process p, when single_gpu is active, is ( p + rotate_gpu ) % Ngpus. Ngpus being the number of GPUs per numa node"});
    ADD_SLOT( long     , smem_bksize , INPUT , OPTIONAL , DocString{"configures shared memory bank size. can be 4 or 8"} );
    ADD_SLOT( DeviceLimitsMap , device_limits  , INPUT , OPTIONAL , DocString{"queries or set device limits. see Cuda documentation of cudaDeviceSetLimit for a list of configurable limits. if value is -1, the device limit is queried ond printed instead of being set. if limit name is 'all', all limits are queried."} );
    ADD_SLOT( bool     , enable_cuda , INPUT_OUTPUT , true , DocString{"if set to false, Cuda support is disabled even if support has been compiled in"} );

  public:

    inline void execute () override final
    {
      MPI_Comm comm = *mpi;
      int nprocs = 1;
      int rank = 0;
      MPI_Comm_size(comm,&nprocs);
      MPI_Comm_rank(comm,&rank);

      lout << "=========== "<<ONIKA_CU_NAME_STR<<" ================"<<std::endl;

#     ifdef ONIKA_CUDA_VERSION

      std::shared_ptr<onika::cuda::CudaContext> cuda_ctx = nullptr;

      int n_gpus = 0;
      if( ! onika::cuda::CudaContext::global_gpu_enable() ) *enable_cuda = false;
      if( *enable_cuda )
      {
        ONIKA_CU_CHECK_ERRORS( ONIKA_CU_GET_DEVICE_COUNT(&n_gpus) );
      }
      if( n_gpus <= 0 )
      {
        lout <<"No GPU found"<<std::endl;
        onika::memory::GenericHostAllocator::set_cuda_enabled( false );
      }
      else
      {
        cuda_ctx = std::make_shared<onika::cuda::CudaContext>();
      
        // multiple GPU per MPI process is not supported by now, because it hasn't been tested for years
        // we replace it with a new feature to allocate GPUs only on even MPI processes, to allow for use of both CPU only and GPU accelerated processes
        int gpu_first_device = ( rank + (*rotate_gpu) ) % n_gpus;
        if( *mpi_even_only )
        {
          if( (rank%2) == 0 )
          {
            gpu_first_device = ( (rank/2) + (*rotate_gpu) ) % n_gpus;
            cuda_ctx->m_devices.resize(1);
          }
          else
          {
            gpu_first_device=0;
            n_gpus=0;
            cuda_ctx->m_devices.clear();
          }
        }
        else
        {
          cuda_ctx->m_devices.resize(1);
        }

        int ndev = cuda_ctx->m_devices.size();

        ldbg <<"ndev="<<ndev<<std::endl;
        for(int d=0;d<ndev;d++) cuda_ctx->m_devices[d].device_id = gpu_first_device + d;
        
        const int max_threads = omp_get_max_threads();
        if( max_threads < ndev )
        {
          fatal_error()<<"Unsupported configuration: number of threads ("<<max_threads<<") less than number of GPUs ("<<ndev<<")"<<std::endl;
        }
        ldbg << "support for a maximum of "<<max_threads<<" threads accessing "<<ndev<<" GPUs"<<std::endl;
        //cuda_ctx->m_threadStream.resize( ndev , 0 );
        //assert( ndev > 0 );
        
        if( ndev > 0 )
        {
          ONIKA_CU_CHECK_ERRORS( ONIKA_CU_SET_DEVICE( cuda_ctx->m_devices[0].device_id ) );

          if( smem_bksize.has_value() )
          {
	    lerr << "smem_bksize is deprecated and has no effect anymore, please remove this setting" << std::endl;
/*
	    switch( *smem_bksize )
            {
              case 4 : ONIKA_CU_CHECK_ERRORS(  ONIKA_CU_SET_SHARED_MEM_CONFIG( onikaSharedMemBankSizeFourByte ) ); break;
              case 8 : ONIKA_CU_CHECK_ERRORS(  ONIKA_CU_SET_SHARED_MEM_CONFIG( onikaSharedMemBankSizeEightByte ) ); break;
              default:
                lerr<<"Unsupported shared memory bank size "<<*smem_bksize<<", using default\n";
                ONIKA_CU_CHECK_ERRORS(  ONIKA_CU_SET_SHARED_MEM_CONFIG( onikaSharedMemBankSizeDefault ) );
                break;
            }
*/
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
            }
            for( const auto& dl : m )
            {
              onikaLimit_t limit;
                   if( dl.first == "cudaLimitStackSize"                    ) limit = onikaLimitStackSize;
              else if( dl.first == "cudaLimitPrintfFifoSize"               ) limit = onikaLimitPrintfFifoSize;
              else if( dl.first == "cudaLimitMallocHeapSize"               ) limit = onikaLimitMallocHeapSize;
              else
              {
                fatal_error() << "Cuda unknown limit '"<<dl.first<<"'"<<std::endl;
              }
              long in_value = std::stol( dl.second );
              
              if( in_value >= 0 )
              {
                ONIKA_CU_CHECK_ERRORS( ONIKA_CU_SET_LIMIT( limit , in_value ) ); 
              }
              else
              {
                size_t value = 0;
                ONIKA_CU_CHECK_ERRORS( ONIKA_CU_GET_LIMIT ( &value, limit ) );
                lout << dl.first << " = " << value << std::endl;
              }
            }
          }
        } // if has device(s)
        
        int n_support_vmm = 0;
        long long totalGlobalMem = 0;
        int warpSize = 32;
        int multiProcessorCount = 0;
        int sharedMemPerBlock = 0;
        int clock_rate = 0;
        int l2_cache = 0;
        std::string device_name = "no-device";

        for(int i=0;i<ndev;i++)
        {
          ONIKA_CU_CHECK_ERRORS( ONIKA_CU_GET_DEVICE_PROPERTIES( & cuda_ctx->m_devices[i].m_deviceProp , i + gpu_first_device ) );
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
#         if ( ! defined(ONIKA_HIP_VERSION) ) || ( HIP_VERSION >= 60000000 )
	  l2_cache = cuda_ctx->m_devices[i].m_deviceProp.persistingL2CacheMaxSize;
#         endif
        }

        long long tmp[3];
        onika::ValueStreamer<long long>(tmp) << ndev << n_support_vmm << totalGlobalMem;
        MPI_Allreduce(MPI_IN_PLACE,tmp,3,MPI_LONG_LONG,MPI_SUM,*mpi);
        onika::ValueStreamer<long long>(tmp) >> ndev >> n_support_vmm >> totalGlobalMem;

        if( n_support_vmm != ndev )
        {
          lerr<<"GPUs don't support unified memory, cannot continue"<<std::endl;
          std::abort();
        }
        
        lout <<"GPUs : "<<ndev<< std::endl;
        lout <<"Type : "<<device_name << std::endl;
        lout <<"SMs  : "<<multiProcessorCount<<"x"<<warpSize<<" threads @ "<< std::defaultfloat<< clock_rate/1000000.0<<" Ghz" << std::endl;

        if( ndev > 0 )
        {
          onika::memory::GenericHostAllocator::set_cuda_enabled( true );
          lout <<"Mem  : "<< onika::memory_bytes_string(totalGlobalMem/ndev) <<" (shared="<<onika::memory_bytes_string(sharedMemPerBlock,"%g%s")<<" L2="<<onika::memory_bytes_string(l2_cache,"%g%s")<<")" <<std::endl;
        }
        else
        {
          cuda_ctx = nullptr;
        }
      }
      
      set_global_cuda_ctx( cuda_ctx );
            
#     else
      lout <<ONIKA_CU_NAME_STR << " disabled"<<std::endl;
#     endif
      lout << "================================="<<std::endl<<std::endl;
    }
  };
  
  // === register factories ===  
  ONIKA_AUTORUN_INIT(init_cuda)
  {
   OperatorNodeFactory::instance()->register_factory( "init_cuda", make_compatible_operator< InitCuda > );
  }

}


