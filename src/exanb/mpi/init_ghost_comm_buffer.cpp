#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/operator_task.h>

#include <exanb/mpi/ghosts_comm_scheme.h>

namespace exanb
{
  

  struct InitGhostCommBuffer : public OperatorNode
  {
    // -----------------------------------------------
    // Operator slots
    // -----------------------------------------------
    ADD_SLOT( GhostCommSendBuffers     , ghost_comm_send_buf , INPUT_OUTPUT , GhostCommSendBuffers{} );
    ADD_SLOT( GhostCommReceiveBuffers  , ghost_comm_recv_buf , INPUT_OUTPUT , GhostCommReceiveBuffers{} );
    
    inline void generate_tasks () override final
    {
      onika_operator_task( ghost_comm_send_buf , ghost_comm_recv_buf )
      {
        ghost_comm_send_buf.m_send_cell_info.clear();
        ghost_comm_send_buf.m_send_buffer.clear();
        ghost_comm_send_buf.m_send_requests.clear();
        
        ghost_comm_recv_buf.m_receive_buffers.clear();
        ghost_comm_recv_buf.m_receive_requests.clear();
      };
    }
    
  };

  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "init_ghost_comm_buffer", make_compatible_operator< InitGhostCommBuffer > );
  }

}

