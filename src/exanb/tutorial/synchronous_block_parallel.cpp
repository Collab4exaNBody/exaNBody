#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/parallel/block_parallel_for.h>

#include "array2d.h"
#include "block_parallel_value_add_functor.h"

namespace tutorial
{
  using namespace exanb;

  class SynchronousBlockParallelSample : public OperatorNode
  {
    ADD_SLOT(Array2D, my_array, INPUT_OUTPUT, REQUIRED);
    ADD_SLOT(double, my_value, INPUT, 1.0);
    
    public:
    
    inline void execute() override final
    {
      using onika::parallel::block_parallel_for;
      
      if( my_array->rows() == 0 || my_array->columns() == 0 )
      {
        my_array->resize( 1024 , 1024 );
      }
    
      BlockParallelValueAddFunctor value_add_func = { *my_array // refernce our data array through its pointer and size
                                                    , *my_value // value to add to the elements of the array
                                                    };
                             
      // Launching the parallel operation, which can execute on GPU if the execution context allows
      block_parallel_for( my_array->rows()             // number of iterations, parallelize at the first level over rows
                        , value_add_func               // the function to call in parallel
                        , parallel_execution_context() // returns the parallel execution context associated with this component
                        );
    }
  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(synchronous_block_parallel)
  {
   OperatorNodeFactory::instance()->register_factory( "synchronous_block_parallel", make_simple_operator< SynchronousBlockParallelSample > );
  }

}

