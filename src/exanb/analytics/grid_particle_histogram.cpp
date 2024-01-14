#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/analytics/grid_particle_histogram.h>

#include <mpi.h>

namespace exaStamp
{

  template<class GridT>
  struct GridParticleHistogram : public OperatorNode
  {
    using ValueType = typename onika::soatl::FieldId<HistField>::value_type ;
      
    ADD_SLOT( MPI_Comm   , mpi       , INPUT , REQUIRED );
    ADD_SLOT( GridT      , grid      , INPUT , REQUIRED );
    ADD_SLOT( long       , resolution   , INPUT , 1024 );
    ADD_SLOT( StringList , fields     , INPUT , StringList({".*"}) , DocString{"List of regular expressions to select fields to slice"} );
    ADD_SLOT( StringMap  , plot_names , INPUT , StringMap{} , DocString{"map field names to output plot names"} );
    ADD_SLOT( Plot1DSet  , plots      , INPUT_OUTPUT );

    inline void execute () override final
    {
      static constexpr onika::soatl::FieldId<HistField> hist_field{};
    }
  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "grid_particle_histogram" , make_grid_variant_operator< GridParticleHistogram > );
  }

}

