#pragma once

#include <onika/soatl/field_tuple.h>
#include <exanb/field_sets.h>
#include <onika/cuda/cuda.h>
#include <onika/memory/allocator.h>
#include <vector>
#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/block_parallel_for.h>
#include <exanb/mpi/ghosts_comm_scheme.h>
#include <exanb/core/yaml_enum.h>


EXANB_YAML_ENUM( exanb , MpiMultipleWaitMethod , EXANB_MPI_MULTIPLE_WAIT , EXANB_MPI_WAIT_ALL , EXANB_MPI_WAIT_ANY );

namespace exanb
{

  struct MpiRequestArray
  {
    MpiMultipleWaitMethod m_wait_method = { MpiMultipleWaitMethod::EXANB_MPI_WAIT_ANY };
    std::vector<MPI_Request> m_requests;
    std::vector<int> m_request_ids;
    std::vector<int> m_completed_request_ids;

    inline void initialize(int capacity)
    {
      m_requests.clear();
      m_requests.reserve(capacity);
      m_request_ids.clear();
      m_request_ids.reserve(capacity);
      m_completed_request_ids.clear();
      m_completed_request_ids.reserve(capacity);
    }

    inline MPI_Request & add_request( MPI_Request req , int req_id )
    {
      m_request_ids.push_back( req_id );
      return m_requests.emplace_back( req );
    }

    inline bool empty() const
    {
      return m_requests.empty();
    }
    
    inline size_t number_of_requests() const
    {
      return m_requests.size();
    }

    inline std::pair<int,const int*> wait_some()
    {
      if( m_requests.size() != m_request_ids.size() )
      {
        fatal_error() << "Inconsistent vector sizes between m_requests and m_request_ids in MpiRequestArray"<<std::endl;
      }
      
      m_completed_request_ids.clear();
      if( m_requests.empty() ) return { 0 , nullptr };

      switch( m_wait_method.value() )
      {
        case MpiMultipleWaitMethod::EXANB_MPI_WAIT_ANY :
        {
          if( m_requests.size() == 1 )
          {
            MPI_Wait( m_requests.data() , MPI_STATUS_IGNORE );
            m_completed_request_ids.push_back( m_request_ids[0] );
            m_requests.clear();
            m_request_ids.clear();
          }
          else
          {
            int reqidx = -1;
            MPI_Waitany( m_requests.size() , m_requests.data() , &reqidx , MPI_STATUS_IGNORE );
            if( reqidx < 0 || reqidx >= ssize_t(m_requests.size()) )
            {
              fatal_error() << "Invalid completed request index "<<reqidx<<std::endl;
            }
            m_completed_request_ids.push_back( m_request_ids.back() );
            m_requests[reqidx] = m_requests.back();
            m_requests.pop_back();
            m_request_ids[reqidx] = m_request_ids.back();
            m_request_ids.pop_back();
          }
          return { 1 , m_completed_request_ids.data() };
        }
        break;
        
        case MpiMultipleWaitMethod::EXANB_MPI_WAIT_ALL :
        {
          MPI_Waitall( m_requests.size() , m_requests.data() , MPI_STATUS_IGNORE );
          m_completed_request_ids = std::move( m_request_ids );
          return { m_completed_request_ids.size() , m_completed_request_ids.data() };
        }
        break;

        case MpiMultipleWaitMethod::EXANB_MPI_MULTIPLE_WAIT :
        {
          for( auto & req : m_requests ) { MPI_Wait( & req , MPI_STATUS_IGNORE ); }
          m_completed_request_ids = std::move( m_request_ids );
          return { m_completed_request_ids.size() , m_completed_request_ids.data() };
        }
        break;
        
        default :
        {
          fatal_error() << "Unknown wait method" << std::endl;
        }
        break;
      }
      
      return { 0 , nullptr } ;
    }
  };

}


