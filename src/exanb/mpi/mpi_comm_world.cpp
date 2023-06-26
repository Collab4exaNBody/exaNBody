#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>

#include <mpi.h>

namespace exanb
{
  

  struct MpiCommWorld : public OperatorNode
  {
    ADD_SLOT( MPI_Comm , mpi , OUTPUT , MPI_COMM_WORLD );
    inline void execute () override final
    {
      assert( *mpi == MPI_COMM_WORLD ) ;
    }
  };

  // === register factory ===
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "mpi_comm_world", make_compatible_operator<MpiCommWorld> );
  }

}

