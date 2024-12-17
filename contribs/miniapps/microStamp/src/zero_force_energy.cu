/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
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
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( double& fx, double& fy, double& fz ) const
    {
      fx = 0.0;
      fy = 0.0;
      fz = 0.0;
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
    , class = AssertGridHasFields< GridT, field::_fx, field::_fy, field::_fz >
    >
  class ZeroForceEnergy : public OperatorNode
  {
    static constexpr FieldSet<field::_fx, field::_fy, field::_fz> zero_field_set{};
  
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
  ONIKA_AUTORUN_INIT(zero_force_energy)
  {
   OperatorNodeFactory::instance()->register_factory( "zero_force_energy", make_grid_variant_operator< ZeroForceEnergyTmpl > );
  }

}

