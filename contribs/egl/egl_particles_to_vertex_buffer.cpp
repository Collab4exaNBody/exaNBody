
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

#include <exanb/core/make_grid_variant_operator.h>

#include <EGLRender/egl_render_manager.h>

namespace exanb
{
  using namespace EGLRender;

  template<class GridT>
  class EGLParticlesToVertexBuffer : public OperatorNode
  {
    ADD_SLOT( GridT            , grid               , INPUT_OUTPUT , DocString{"Local sub-domain particles grid"} );
    ADD_SLOT( std::string      , vertex_buffer      , INPUT_OUTPUT , "particles" );
    ADD_SLOT( EGLRenderManager , egl_render_manager , INPUT_OUTPUT );

  public:
    inline void execute() override final
    {
      const size_t n_points = grid->number_of_particles() - grid->number_of_ghost_particles();

      int buf_id = egl_render_manager->vertex_buffers_id( *vertex_buffer );
      if( buf_id == -1 )
      {
        ldbg << "EGL : create vertex buffer " << *vertex_buffer <<std::endl;
        const GLint attrib_formats[] = { GL_FLOAT,3 };
        buf_id = egl_render_manager->create_vertex_buffers( *vertex_buffer , n_points , attrib_formats );
      }
      auto & glvbos = egl_render_manager->vertex_buffers(buf_id);
      ldbg << "EGL : update vertex buffer " << *vertex_buffer << " , nv="<< n_points << " , id="<<buf_id<<std::endl;

      if( glvbos.number_of_attribs() != 1 || glvbos.attrib_type(0)!=GL_FLOAT || glvbos.attrib_components(0)!=3 )
      {
        onika::fatal_error() << "Works only with 1 vertex attribute : attribute 0 with format GLfloat x3" << std::endl;
      }

      glvbos.set_number_of_vertices( n_points );

      GLfloat* v = (GLfloat*) glvbos.map_buffer_write_only(0);

      const size_t n_cells = grid->number_of_cells();
      const auto cells = grid->cells_accessor();
      size_t vertex_idx = 0;
      for(size_t c=0;c<n_cells;c++)
      {
        if( ! grid->is_ghost_cell(c) )
        {
          const size_t n_cell_particles = cells[c].size();
          const auto rx = cells[c][field::rx];
          const auto ry = cells[c][field::ry];
          //const auto rz = cells[c][field::rz];
          for( size_t p=0 ; p<n_cell_particles ; p++ , vertex_idx++ )
          {
            v[ vertex_idx*3 + 0 ] = rx[p]/50.0 - 0.5;
            v[ vertex_idx*3 + 1 ] = ry[p]/50.0 - 0.5;
            v[ vertex_idx*3 + 2 ] = 0.0; //rz[p];
          }
        }
      }

      glvbos.unmap_buffer(0);
    }

  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(egl_particles_to_vertex_buffer)
  {
    OperatorNodeFactory::instance()->register_factory( "egl_particles_to_vertex_buffer", make_grid_variant_operator< EGLParticlesToVertexBuffer > );
  }

}

