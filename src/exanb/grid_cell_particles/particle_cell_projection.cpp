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
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/grid_cell_particles/grid_cell_values.h>

#include <exanb/grid_cell_particles/particle_cell_projection.h>
#include <exanb/core/grid_particle_field_accessor.h>

#include <regex>

//#include <exanb/compute/math_functors.h>
// allow field combiner to be processed as standard field
//ONIKA_DECLARE_FIELD_COMBINER( exanb, VelocityNorm2Combiner , vnorm2 , exanb::Vec3Norm2Functor , exanb::field::_vx , exanb::field::_vy , exanb::field::_vz )

namespace exanb
{

  template< class GridT >
  class ParticleCellProjection : public OperatorNode
  {
    using StringList = std::vector<std::string>;
    
    ADD_SLOT( GridT          , grid              , INPUT , REQUIRED );
    ADD_SLOT( double         , splat_size        , INPUT , REQUIRED );
    ADD_SLOT( long           , grid_subdiv       , INPUT , 1 );
    ADD_SLOT( GridCellValues , grid_cell_values  , INPUT );
    ADD_SLOT( StringList  , fields            , INPUT , StringList({".*"}) , DocString{"List of regular expressions to select fields to project"} );

  public:

    // -----------------------------------------------
    inline void execute ()  override final
    {
      using namespace ParticleCellProjectionTools;
      if( grid->number_of_cells() == 0 ) return;

      const auto& flist = *fields;
      auto field_selector = [&flist] ( const std::string& name ) -> bool { for(const auto& f:flist) if( std::regex_match(name,std::regex(f)) ) return true; return false; } ;

      // create cell value fields
      std::vector<AddCellFieldInfo> fields_to_add;
      CollectCellValueFieldToAdd collect_fields = {*grid_cell_values,fields_to_add,field_selector,*grid_subdiv};
      apply_grid_field_set( *grid, collect_fields , GridT::field_set );
      grid_cell_values->add_fields( fields_to_add );

      // project particle quantities to cells
      using ParticleAcessor = GridParticleFieldAccessor<typename GridT::CellParticles *>;
      ProjectCellValueField<ParticleAcessor> project_fields = { {grid->cells()} , *grid_cell_values,field_selector,*splat_size,*grid_subdiv};
      apply_grid_field_set( *grid, project_fields , GridT::field_set );
    }

    // -----------------------------------------------
    // -----------------------------------------------
    inline std::string documentation() const override final
    {
      return R"EOF(project particle quantities onto a regular grid)EOF";
    }    

  };

  // === register factories ===
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory("particle_cell_projection", make_grid_variant_operator< ParticleCellProjection > );
  }

}
