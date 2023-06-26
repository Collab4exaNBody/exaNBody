#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/log.h>
#include <exanb/core/domain.h>
#include <exanb/core/string_utils.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/simple_block_rcb.h>
#include <exanb/core/parallel_random.h>
#include <exanb/core/grid_algorithm.h>

#include <mpi.h>
#include <algorithm>

namespace exanb
{
  

  namespace LoadBalanceRandom_details
  {

    GridBlock rcb_block_split_cuts( GridBlock block, size_t n_parts, size_t part, const double* cuts )
    {
      if( n_parts <= 1 ) { return block; }
      assert( part>=0 && part<n_parts );
      size_t pivot = n_parts/2;
      bool side = ( part >= pivot );
      IJK dims = dimension(block);
      
      if( dims.i >= dims.j && dims.i >= dims.k )
      {
        assert( dims.i >= 2 );
        ssize_t s = block.start.i + 1 + std::clamp( static_cast<int>( (dims.i-1) * (*cuts) ) , 0, static_cast<int>(dims.i-2) );
        assert( s > block.start.i && s < block.end.i );
        if( side ) { block.start.i = s; }
        else       { block.end.i   = s; }
      }
      else if( dims.j >= dims.i && dims.j >= dims.k )
      {
        assert( dims.j >= 2 );
        ssize_t s = block.start.j + 1 + std::clamp( static_cast<int>( (dims.j-1) * (*cuts) ) , 0, static_cast<int>(dims.j-2 ) );
        assert( s > block.start.j && s < block.end.j );
        if( side ) { block.start.j = s; }
        else       { block.end.j   = s; }
      }
      else
      {
        assert( dims.k >= 2 );
        ssize_t s = block.start.k + 1 + std::clamp( static_cast<int>( (dims.k-1) * (*cuts) ) , 0, static_cast<int>(dims.k-2 ) );
        assert( s > block.start.k && s < block.end.k );
        if( side ) { block.start.k = s; }
        else       { block.end.k   = s; }
      }

      size_t sub_group_size = n_parts/2;
      size_t sub_group_rank = part;
      if( side )
      {
        sub_group_size = n_parts - (n_parts/2);
        sub_group_rank = part - (n_parts/2);
      }

      return rcb_block_split_cuts( block , sub_group_size , sub_group_rank , cuts+1 );
    }

  }
  
  using namespace LoadBalanceRandom_details;

  struct LoadBalanceRandom : public OperatorNode
  {
    ADD_SLOT( MPI_Comm  , mpi      , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( Domain    , domain   , INPUT );
    ADD_SLOT( GridBlock , lb_block , INPUT_OUTPUT );
    ADD_SLOT( double    , lb_inbalance, INPUT_OUTPUT);

    inline void execute () override final
    {
      int size = 1;      
      int rank = 0;
      MPI_Comm_rank(*mpi,&rank);
      MPI_Comm_size(*mpi,&size);

      std::vector<double> cuts(size);
      if( rank == 0 )
      {
        auto& re = rand::random_engine();
        std::uniform_real_distribution<double> uniform01( 0.0 , 1.0 );
        for(int i=0;i<size;i++)
        {
          cuts[i] = uniform01(re);
        }
      }
      MPI_Bcast(cuts.data(),size,MPI_DOUBLE,0,*mpi);

      GridBlock in_block = { IJK{0,0,0} , domain->grid_dimension() };
      *lb_block = rcb_block_split_cuts( in_block, size, rank, cuts.data() );

      ldbg << "LoadBalanceRandom: in_block="<< in_block << ", out_block=" << *lb_block << std::endl;
      *lb_inbalance = 99.99;
    }
        
  };

  // === register factory ===
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory(
      "load_balance_random",
      make_compatible_operator<LoadBalanceRandom> );
  }

}

