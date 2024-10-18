#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <onika/parallel/block_parallel_for.h>

#include "array2d.h"
#include "block_parallel_value_add_functor.h"

namespace tutorial
{
  using namespace exanb;

  class AsyncBlockParallelSample : public OperatorNode
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
      // result of parallel operation construct is captured into variable 'my_addition',
      // thus it can be scheduled in a stream queue for asynchronous execution rather than being executed right away
      auto my_addition = block_parallel_for( my_array->rows() // number of iterations, parallelize at the first level over rows
                                           , value_add_func   // the function to call in parallel
                                           , parallel_execution_context("my_add_kernel") // execution environment inherited from this OperatorNode
                                           ); // optionally, we may tag here ^^^ parallel operation for debugging/profiling purposes
      // my_addition is scheduled here, transfering its content/ownership (see std::move) to the default stream queue
      auto stream_control = parallel_execution_stream() << std::move(my_addition) ;
      lout << "Parallel operation is executing..." << std::endl;
      stream_control.wait();                               // wait for the operation to complete and results to be ready to read
      lout << "Parallel operation has completed !" << std::endl;
    }
  };

  // === register factories ===
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "async_block_parallel", make_simple_operator< AsyncBlockParallelSample > );
  }

}

