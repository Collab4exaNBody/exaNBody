#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/grid.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_random.h>

#include <exanb/grid_cell_particles/particle_region.h>
#include <exanb/compute/gaussian_noise.h>

namespace microStamp
{
  using namespace exanb;

  template<class GridT> using GaussianNoiseR = GaussianNoise < GridT , field::_id , FieldSet<field::_rx,field::_ry,field::_rz> >;
  template<class GridT> using GaussianNoiseV = GaussianNoise < GridT , field::_id , FieldSet<field::_vx,field::_vy,field::_vz> >;
  template<class GridT> using GaussianNoiseF = GaussianNoise < GridT , field::_id , FieldSet<field::_fx,field::_fy,field::_fz> >;

  // === register factories ===
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "gaussian_noise_r", make_grid_variant_operator< GaussianNoiseR > );
   OperatorNodeFactory::instance()->register_factory( "gaussian_noise_v", make_grid_variant_operator< GaussianNoiseV > );
   OperatorNodeFactory::instance()->register_factory( "gaussian_noise_f", make_grid_variant_operator< GaussianNoiseF > );
  }

}

