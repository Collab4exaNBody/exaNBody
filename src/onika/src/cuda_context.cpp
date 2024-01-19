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


