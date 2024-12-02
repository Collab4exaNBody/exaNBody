#pragma xstamp_cuda_enable

#pragma xstamp_grid_variant

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>

#include <onika/cuda/cuda.h>
#include <exanb/compute/compute_cell_particles.h>


namespace microStamp
{
  using namespace exanb;
  using namespace onika;

  struct ZeroCellParticleFields
  {
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( double& fx, double& fy, double& fz, double& ep ) const
    {
      fx = 0.0;
      fy = 0.0;
      fz = 0.0;
      ep = 0.0;
    }
  };
}

namespace exanb
{
  template<> struct ComputeCellParticlesTraits<microStamp::ZeroCellParticleFields>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = true;
  };
}

namespace microStamp
{
  using namespace exanb;

  template<typename GridT
    , class = AssertGridHasFields< GridT, field::_fx, field::_fy, field::_fz, field::_ep >
    >
  class ZeroForceEnergy : public OperatorNode
  {
    static constexpr FieldSet<field::_fx, field::_fy, field::_fz,field::_ep> zero_field_set{};
  
    ADD_SLOT( GridT , grid  , INPUT_OUTPUT );
    ADD_SLOT( bool  , ghost  , INPUT , false );

  public:

    inline void execute () override final
    {
      ZeroCellParticleFields zero_op = {};
      compute_cell_particles( *grid , *ghost , zero_op , zero_field_set , parallel_execution_context() );
    }

  };
  
  template<class GridT> using ZeroForceEnergyTmpl = ZeroForceEnergy<GridT>;
  
 // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "zero_force_energy", make_grid_variant_operator< ZeroForceEnergyTmpl > );
  }

}

