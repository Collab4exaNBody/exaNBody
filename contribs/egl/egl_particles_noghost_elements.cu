
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
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>
#include <onika/log.h>
#include <onika/math/basic_types_def.h>
#include <onika/math/quaternion.h>

#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/compute/field_combiners.h>
#include <exanb/compute/compute_cell_particles.h>
#include <exanb/core/particle_type_properties.h>
#include <exanb/core/grid_additional_fields.h>

#include <EGLRender/egl_render_manager.h>

namespace exanb
{
  using namespace EGLRender;

  using Vec3d = exanb::Vec3d;
  using Quat = exanb::Quaternion;
  using Mat3d = exanb::Mat3d;

  template<class GridT>
  class EGLParticlesNoGhostElements : public OperatorNode
  {
    ADD_SLOT( GridT            , grid               , INPUT_OUTPUT , DocString{"Local sub-domain particles grid"} );
    ADD_SLOT( std::string      , elements , INPUT_OUTPUT , "noghost_particles" );
    ADD_SLOT( EGLRenderManager , egl_render_manager , INPUT_OUTPUT );

  public:

    inline void execute() override final
    {
      const auto particles_noghost = grid->number_of_particles() - grid->number_of_ghost_particles();
      const auto elbuf_id = egl_render_manager->create_element_buffer( *elements , particles_noghost );

      auto * elptr = egl_render_manager->element_buffer(elbuf_id).map_buffer_write_only();
      size_t element_count = 0;
      const auto * cell_particle_offset = grid->cell_particle_offset_data();
      
      IJK dims = grid->dimension();
      ssize_t gl = grid->ghost_layers();
      IJK dimsNoGhost = { dims.i-2*gl , dims.j-2*gl , dims.k-2*gl };
      for(ssize_t k=0;k<dimsNoGhost.k;k++)
      for(ssize_t j=0;j<dimsNoGhost.j;j++)
      for(ssize_t i=0;i<dimsNoGhost.i;i++)
      {
        ssize_t cell_i = grid_ijk_to_index(dims,IJK{i+gl,j+gl,k+gl});
        const int n_particles = grid->cell_number_of_particles(cell_i);
        for(int p=0;p<n_particles;p++)
        {
          elptr[element_count++] = cell_particle_offset[cell_i] + p;
        }
      }
      
      assert( element_count == egl_render_manager->element_buffer(elbuf_id).size() );
      egl_render_manager->element_buffer(elbuf_id).unmap_buffer();
    }

  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(egl_particles_noghost_elements)
  {
    OperatorNodeFactory::instance()->register_factory( "egl_particles_noghost_elements", make_grid_variant_operator< EGLParticlesNoGhostElements > );
  }

}

