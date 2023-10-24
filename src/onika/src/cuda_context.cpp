#include <onika/cuda/cuda_context.h>

// specializations to avoid MemoryUsage template to dig into cuda aggregates
namespace onika
{

  namespace cuda
  {

    bool CudaContext::has_devices() const
    {
      return ! m_devices.empty();
    }
    
    unsigned int CudaContext::device_count() const
    {
      return m_devices.size();
    }
    
    cudaStream_t CudaContext::getThreadStream(unsigned int tid)
    {
      if( tid >= m_threadStream.size() )
      {
        unsigned int i = m_threadStream.size();
        m_threadStream.resize( tid+1 , 0 );
        for(;i<m_threadStream.size();i++)
        {
          std::cout << "create stream #"<<i<<std::endl;
          checkCudaErrors( ONIKA_CU_CREATE_STREAM_NON_BLOCKING( m_threadStream[i] ) );
          // cudaStreamCreateWithFlags( & m_threadStream[i], cudaStreamNonBlocking );
        }
      }
      return m_threadStream[tid];
    }

  }

}


