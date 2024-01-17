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
    
    onikaStream_t CudaContext::getThreadStream(unsigned int tid)
    {
      if( tid >= m_threadStream.size() )
      {
        unsigned int i = m_threadStream.size();
        m_threadStream.resize( tid+1 , 0 );
        for(;i<m_threadStream.size();i++)
        {
          ONIKA_CU_CHECK_ERRORS( ONIKA_CU_CREATE_STREAM_NON_BLOCKING( m_threadStream[i] ) );
        }
      }
      return m_threadStream[tid];
    }

  }

}


