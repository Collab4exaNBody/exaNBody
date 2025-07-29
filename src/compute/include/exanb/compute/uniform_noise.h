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
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <onika/parallel/random.h>

#include <exanb/grid_cell_particles/particle_localized_filter.h>

namespace exanb
{

  template<class RNG, class CellT, class ParticleFilterFuncT, class IdField, class... field_ids>
  static inline void apply_uniform_noise( CellT& cell, RNG& re, double sigma,
            const ParticleFilterFuncT& particle_filter,
            IdField ID,
            FieldSet<field_ids...> )
  {
    static constexpr bool has_id_field = CellT::has_field( ID );
    std::uniform_real_distribution<double> uniform(-sigma,sigma);
    const size_t n = cell.size();
    for(size_t i=0;i<n;i++)
    {
      Vec3d r = { cell[field::rx][i] , cell[field::ry][i] , cell[field::rz][i] };
      uint64_t id = 0;
      if constexpr ( has_id_field ) { id = cell[ID][i]; }
      if( particle_filter(r,id) )
      {
        ( ... , ( cell[onika::soatl::FieldId<field_ids>()] [i] += uniform(re) ) );
      }
    }
  }

  template<
    class GridT,
    class IdField,
    class FieldSetT,
    class = AssertGridContainFieldSet< GridT, FieldSetT >
    >
  class UniformNoise : public OperatorNode
  {
    ADD_SLOT( GridT  , grid    , INPUT_OUTPUT );
    ADD_SLOT( Domain , domain  , INPUT );

    ADD_SLOT( double , sigma   , INPUT , 1.0 , DocString("(YAML: float) Min/max value of the uniform (centered) distribution"));
    ADD_SLOT( bool   , ghost   , INPUT , false );

    // optionaly limit noise to a geometric region
    ADD_SLOT( ParticleRegions   , particle_regions , INPUT , OPTIONAL , DocString("(YAML: boolean expression) Boolean expression of particle_regions in which the gaussian noise should be applied."));
    ADD_SLOT( ParticleRegionCSG , region           , INPUT_OUTPUT , OPTIONAL );

    // optionaly limit lattice generation to places where some mask has some value
    ADD_SLOT( GridCellValues , grid_cell_values    , INPUT , OPTIONAL );
    ADD_SLOT( std::string    , grid_cell_mask_name , INPUT , OPTIONAL , DocString("(YAML: string) Name of grid_cell_values mask that will act as a mask region in which the gaussian noise should be applied.") );
    ADD_SLOT( double         , grid_cell_mask_value, INPUT , OPTIONAL , DocString("(YAML: float) Value of the grid_cell_values used to define the mask.") );

    static constexpr onika::soatl::FieldId<IdField> field_id = {};
    static constexpr FieldSetT field_set = {};

  public:

    // -----------------------------------------------
    // -----------------------------------------------
    inline void execute ()  override final
    {
      PartcileLocalizedFilter<GridT,LinearXForm> particle_filter = { *grid, { domain->xform() } };
      particle_filter.initialize_from_optional_parameters( particle_regions.get_pointer(),
                                                           region.get_pointer(),
                                                           grid_cell_values.get_pointer(), 
                                                           grid_cell_mask_name.get_pointer(), 
                                                           grid_cell_mask_value.get_pointer() );

      auto cells = grid->cells();
      IJK dims = grid->dimension();
      ssize_t gl = 0; 
      if( ! *ghost ) { gl = grid->ghost_layers(); }
      IJK gstart { gl, gl, gl };
      IJK gend = dims - IJK{ gl, gl, gl };
      IJK gdims = gend - gstart;

#     pragma omp parallel
      {
        auto& re = onika::parallel::random_engine();
        GRID_OMP_FOR_BEGIN(gdims,_,loc, schedule(dynamic) )
        {
          size_t i = grid_ijk_to_index( dims , loc + gstart );
          apply_uniform_noise( cells[i], re, *sigma, particle_filter, field_id, field_set );
        }
        GRID_OMP_FOR_END
      }
    }

    // -----------------------------------------------
    // -----------------------------------------------
    inline std::string documentation() const override final
    {
      return R"EOF(
Apply a white gaussian noise to selected fields. In addition, a cutoff can be passed to the operator to avoid values beyond it. Be careful, doing this do not lead to a gaussian distribution but a truncated one!
Note: processes particles in ghost layers if and only if ghost input is true

Usage example:

input_data:
  - particle_types:
      particle_type_map: { A: 0 }
  - particle_type_add_properties:
      A: { mass: 25.0 Da }
  - domain:
      cell_size: 5.0 ang
      grid_dims: [20,20,20]
      bounds: [[0 ang ,0 ang,0 ang],[100 ang, 100 ang, 100 ang]]
      xform: [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
      periodic: [true,true,true]
      expandable: false
  - init_rcb_grid
  - lattice:
      structure: BCC
      types: [ A, A]
      size: [ 5.0 ang , 5.0 ang , 5.0 ang ]
  - uniform_noise_r: { sigma: 0.2 ang }
  - uniform_noise_v: { sigma: 50. ang/ps }
  - uniform_noise_v: { sigma: 50. ang/ps, region: SPHERE }
  - uniform_noise_v: { sigma: 50. ang/ps, grid_cell_mask_name: MYMASK, grid_cell_mask_value: 1. }
)EOF";
    }

  };

}

