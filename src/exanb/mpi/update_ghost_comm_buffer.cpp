#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/log.h>
#include <exanb/core/basic_types.h>
#include <exanb/core/grid.h>
#include <exanb/core/operator_task.h>

#include <mpi.h>
#include <exanb/mpi/update_ghost_utils.h>
#include <exanb/mpi/ghosts_comm_scheme.h>
#include <exanb/mpi/constants.h>

namespace exanb
{
  

  using namespace UpdateGhostsUtils;

  template<class... _FieldIds>
  struct UpdateGhostsSendBuffer : public OperatorNode
  {
    using ParticleTuple = typename FieldSetToParticleTuple< AddDefaultFields< FieldSet<_FieldIds...> > >::type;
    static_assert( sizeof(uint8_t) == 1 , "uint8_t is not a byte");

    // -----------------------------------------------
    // Operator slots
    // -----------------------------------------------
    ADD_SLOT( MPI_Comm                 , mpi                 , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( GhostCommunicationScheme , ghost_comm_scheme   , INPUT_OUTPUT , REQUIRED );
    ADD_SLOT( GhostCommSendBuffers     , ghost_comm_send_buf , INPUT_OUTPUT , REQUIRED );
    ADD_SLOT( GhostCommReceiveBuffers  , ghost_comm_recv_buf , INPUT_OUTPUT , REQUIRED );
    
    inline void generate_tasks () override final
    {

      onika_operator_task( mpi , ghost_comm_scheme , ghost_comm_send_buf , ghost_comm_recv_buf )
      {
        int _nprocs = 1;
        int rank = 0;
        MPI_Comm_size(mpi,&_nprocs);
        MPI_Comm_rank(mpi,&rank);
        unsigned int nprocs = _nprocs;

        // bytes per particle can be updated once we know what fields are to be updated
        if( ghost_comm_scheme.m_particle_bytes == 0 )
        {
          ghost_comm_scheme.m_particle_bytes = sizeof( ParticleTuple );
        }

        IJK dims = ghost_comm_scheme.m_grid_dims;
        size_t cell_bytes = ghost_comm_scheme.m_cell_bytes;
        size_t particle_bytes = ghost_comm_scheme.m_particle_bytes;
        assert( nprocs == ghost_comm_scheme.m_partner.size() );

        ghost_comm_send_buf.m_send_cell_info.clear();

        ghost_comm_send_buf.m_send_buffer.resize( nprocs );
        ghost_comm_send_buf.m_send_requests.assign( nprocs , MPI_REQUEST_NULL );
        
        ghost_comm_recv_buf.m_receive_buffers.resize( nprocs );
        ghost_comm_recv_buf.m_receive_requests.assign( nprocs , MPI_REQUEST_NULL );

        for(unsigned int p=0;p<nprocs;p++)
        {
          size_t bufsize = 0;
          auto& partner = ghost_comm_scheme.m_partner[p];
          unsigned int n_send_cells = partner.m_sends.size();          
          for(unsigned int i=0;i<n_send_cells;i++)
          {
            IJK loc = grid_index_to_ijk( dims , partner.m_sends[i].m_cell_i );
            ghost_comm_send_buf.m_send_cell_info.emplace( std::make_pair( loc , GhostSendCellInfo { p , i , bufsize } ) );
            bufsize += cell_bytes + partner.m_sends[i].m_particle_i.size() * particle_bytes;
          }
          // lout_stream() << "P"<<rank<<" has "<<n_send_cells<<" cells and "<<bufsize<<" bytes to send to P"<<p<<std::endl;
          ghost_comm_send_buf.m_send_buffer[p].m_cells_to_send = n_send_cells;
          ghost_comm_send_buf.m_send_buffer[p].m_cell_counter.store( n_send_cells , std::memory_order_release );
          ghost_comm_send_buf.m_send_buffer[p].m_buffer.resize( bufsize );

          unsigned int n_receive_cells = partner.m_receives.size();
          bufsize = 0;
          for(unsigned int i=0;i<n_receive_cells;i++)
          {
            auto [ cell_i , n_particles ] = ghost_cell_receive_info( partner.m_receives[i] );
            bufsize += cell_bytes + n_particles * particle_bytes;
          }
          ghost_comm_recv_buf.m_receive_buffers[p].resize( bufsize );
        }
        
      };
      
    }
    
  };

  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "ghost_update_comm_buffer_r", make_simple_operator< UpdateGhostsSendBuffer<field::_rx,field::_ry,field::_rz> > );
  }

}

