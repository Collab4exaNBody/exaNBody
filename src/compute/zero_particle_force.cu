#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>

#include <onika/cuda/cuda.h>
#include <exanb/compute/compute_cell_particles.h>

namespace exanb
{
  using namespace onika;

  struct ZeroForceFields
  {
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( double& fx, double& fy, double& fz ) const
    {
      fx = 0.0;
      fy = 0.0;
      fz = 0.0;
    }
  };

  template<> struct ComputeCellParticlesTraits<ZeroForceFields>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = true;
  };

  template<class GridT>
  class ZeroParticleForce : public OperatorNode
  {  
    ADD_SLOT( GridT , grid  , INPUT_OUTPUT );
    ADD_SLOT( bool  , ghost  , INPUT , false );

  public:

    inline void execute () override final
    {
      auto fx = grid->field_accessor( field::fx );
      auto fy = grid->field_accessor( field::fy );
      auto fz = grid->field_accessor( field::fz );
      ZeroForceFields func = {};
      compute_cell_particles( *grid , *ghost , func, onika::make_flat_tuple(fx,fy,fz) , parallel_execution_context() );
    }

  };
    
 // === register factories ===  
  ONIKA_AUTORUN_INIT(zero_particle_force)
  {
   OperatorNodeFactory::instance()->register_factory( "zero_particle_force", make_grid_variant_operator< ZeroParticleForce > );
  }

}

