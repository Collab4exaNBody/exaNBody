#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/log.h>
#include <exanb/core/domain.h>
#include <exanb/core/string_utils.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/simple_block_rcb.h>

#include <mpi.h>

namespace exanb
{
  

  using std::vector;
  using std::string;
  using std::endl;

  struct LoadBalanceUniformCostsRCBNode : public OperatorNode
  {
    ADD_SLOT( MPI_Comm  , mpi      , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( Domain    , domain   , INPUT );
    ADD_SLOT( GridBlock , lb_block , INPUT_OUTPUT );
    ADD_SLOT( double    , lb_inbalance, INPUT_OUTPUT);

    inline void execute () override final
    {
      MPI_Comm comm = *mpi;
      Domain domain = *(this->domain);
      GridBlock& out_block = *lb_block;

      int size = 1;      
      int rank = 0;
      MPI_Comm_rank(comm,&rank);
      MPI_Comm_size(comm,&size);

      GridBlock in_block = { IJK{0,0,0} , domain.grid_dimension() };
      out_block = simple_block_rcb( in_block, size, rank );

      ldbg << "LoadBalanceUniformCostsRCBNode: in_block="<< in_block << ", out_block=" << out_block << std::endl;
      *lb_inbalance = 0.1;
    }
        
  };

  // === register factory ===
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory(
      "load_balance_no_costs",
      make_compatible_operator<LoadBalanceUniformCostsRCBNode> );
  }

}

