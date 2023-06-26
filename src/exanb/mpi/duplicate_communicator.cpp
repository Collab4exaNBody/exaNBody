#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>

#include <mpi.h>

namespace exanb
{
  

  struct CommDupOperator : public OperatorNode
  {
    ADD_SLOT( MPI_Comm , mpi     , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( MPI_Comm , mpi_dup , OUTPUT );

    inline void execute () override final
    {
      MPI_Comm c;
      MPI_Comm_dup( *mpi , & c );
      *mpi_dup = c;
    }
  };


  // === register factory ===
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "mpi_dup", make_compatible_operator<CommDupOperator> );
  }

}

