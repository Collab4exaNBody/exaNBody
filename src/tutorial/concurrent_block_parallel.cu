#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/parallel/block_parallel_for.h>

#include "array2d.h"
#include "block_parallel_value_add_functor.h"

namespace tutorial
{
  using namespace exanb;

  class ConcurrentBlockParallelSample : public OperatorNode
  {
    ADD_SLOT(Array2D, array1, INPUT_OUTPUT, REQUIRED);
    ADD_SLOT(Array2D, array2, INPUT_OUTPUT, REQUIRED);
    ADD_SLOT(double, value, INPUT, 1.0);
    
    public:
    
    inline void execute() override final
    {
      using onika::parallel::block_parallel_for;

      // if array1 is empty, allocate it
      if( array1->rows() == 0 || array1->columns() == 0 )
      {
        array1->resize( 1024 , 1024 );
      }

      // if array2 is empty, allocate it
      if( array2->rows() == 0 || array2->columns() == 0 )
      {
        array2->resize( 1024 , 1024 );
      }

      // functor to add the same value to all elements of array1    
      BlockParallelValueAddFunctor array1_add_func = { *array1 // refernce our data array through its pointer and size
                                                     , *value // value to add to the elements of the array
                                                     };

      // functor to add the same value to all elements of array2
      BlockParallelValueAddFunctor array2_add_func = { *array2, *value };
                             
      // Launching the parallel operation, which can execute on GPU if the execution context allows
      // result of parallel operation construct is captured into variable 'my_addition',
      // thus it can be scheduled in a stream queue for asynchronous execution rather than being executed right away
      auto addition1 = block_parallel_for( array1->rows()  // number of iterations, parallelize at the first level over rows
                                         , array1_add_func // the function to call in parallel
                                         , parallel_execution_context("add_kernel") // execution environment inherited from this OperatorNode
                                         ); // optionally, we may tag here ^^^ parallel operation for debugging/profiling purposes

      // we create a second parallel operation we want to execute sequentially after the first addition
      auto addition2 = block_parallel_for( array1->rows(), array1_add_func, parallel_execution_context("add_kernel") );

      // we finally create a third parallel operation we want to execute concurrently with the two others
      auto addition3 = block_parallel_for( array2->rows(), array2_add_func, parallel_execution_context("add_kernel") );

      // we create 2 custom queues with different default execution lane
      auto stream_0_control = parallel_execution_custom_queue(0);
      auto stream_1_control = parallel_execution_custom_queue(1);
      
      // addition1 and addition2 are scheduled asyncronously and sequentially one after the other, in the stream queue #0
      stream_0_control << std::move(addition1) << std::move(addition2);
      
      // addition3 is scheduled asynchrounsly in stream queue #1, thus it may run concurrently with operations in stream quaue #0
      stream_1_control << std::move(addition3) ;
      
      lout << "Parallel operations are executing..." << std::endl;
      stream_0_control.wait(); // wait for all operations in stream queue #0 to complete
      stream_1_control.wait(); // wait for all operations in stream queue #1 to complete
      lout << "All parallel operations have terminated !" << std::endl;
    }
  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(concurrent_block_parallel)
  {
   OperatorNodeFactory::instance()->register_factory( "concurrent_block_parallel", make_simple_operator< ConcurrentBlockParallelSample > );
  }

}

