#pragma once

#include <mpi.h>
#include <cstdint>
#include <mutex>
#include <condition_variable>

namespace exanb
{

  struct ParticleDisplOverAsyncRequest
  {
    MPI_Comm m_comm = MPI_COMM_NULL;
    MPI_Request m_request = MPI_REQUEST_NULL;
    
    unsigned long long m_particles_over = 0;
    unsigned long long m_all_particles_over = 0; 
    
    GPUStreamCallback m_reduction_end_callback = {nullptr,nullptr,nullptr,0};
    
    std::mutex m_request_mutex;
    std::condition_variable m_request_cond;
    bool m_async_request = false;
    bool m_request_started = false;

    inline void start_mpi_async_request()
    {
      assert( m_async_request );
      std::unique_lock lk( m_request_mutex );
      //std::cout<<"start async MPI reduction"<<std::endl;
      MPI_Iallreduce( (const void*) & m_particles_over , (void*) & m_all_particles_over , 1, MPI_UNSIGNED_LONG_LONG , MPI_SUM, m_comm, & m_request );
      m_request_started = true;
      lk.unlock();
      m_request_cond.notify_one();
    }
    
    inline unsigned long long wait()
    {
      if( m_async_request )
      {
        {
          std::unique_lock lk( m_request_mutex );
          m_request_cond.wait(lk, [this]{return m_request_started;});
        }
        //std::cout<<"wait for MPI reduction completion"<<std::endl;
        MPI_Status status;
        MPI_Wait( &m_request, &status );
        m_async_request = false;
        m_request_started = false;
        m_comm = MPI_COMM_NULL;
        m_request = MPI_REQUEST_NULL;
        m_reduction_end_callback = GPUStreamCallback{nullptr,nullptr,nullptr,0};
      }
      assert( m_all_particles_over >= m_particles_over );
      return m_all_particles_over;
    }
    
  };

}
