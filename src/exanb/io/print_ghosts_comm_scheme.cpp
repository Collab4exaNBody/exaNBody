#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/log.h>
#include <exanb/core/basic_types_stream.h>

#include <exanb/mpi/ghosts_comm_scheme.h>

namespace exanb
{
  

  struct PrintGhostsCommScheme : public OperatorNode
  {
    // -----------------------------------------------
    // Operator slots
    // -----------------------------------------------
    ADD_SLOT( GhostCommunicationScheme , ghost_comm_scheme , INPUT_OUTPUT);

    inline void execute () override final
    {
      to_stream( lout , *ghost_comm_scheme );
    }

  };

  // === register factory ===
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory("print_ghost_comm_scheme",make_simple_operator< PrintGhostsCommScheme > );
  }

}

