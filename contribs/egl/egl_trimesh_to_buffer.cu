
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
#include <exanb/mesh/triangle_mesh.h>
#include <EGLRender/egl_render_manager.h>

namespace exanb
{
  using namespace EGLRender;

  class EGLTriMeshToBuffer : public OperatorNode
  {
    using StringIntMap = std::map< std::string , int >;

    ADD_SLOT( std::string      , vertex_buffer      , INPUT_OUTPUT , "mesh_vertices" );
    ADD_SLOT( std::string      , element_buffer      , INPUT_OUTPUT , "mesh_triangles" );
    ADD_SLOT( TriangleMesh     , mesh              , INPUT_OUTPUT , REQUIRED );
    ADD_SLOT( EGLRenderManager , egl_render_manager , INPUT_OUTPUT );

  public:

    inline void execute() override final
    {
      const long vcount = mesh->vertex_count();
      const long tcount = mesh->triangle_count();
      
      
      int vbuf_id = egl_render_manager->vertex_buffers_id( *vertex_buffer );
      if( vbuf_id == -1 )
      {
        ldbg << "EGL : create vertex buffer " << *vertex_buffer <<std::endl;
        std::vector<GLint> vertex_attribs = { GL_FLOAT, 3 };
        vbuf_id = egl_render_manager->create_vertex_buffers( *vertex_buffer , vcount , vertex_attribs );
      }
      
      GLVertexBuffers & glvbos = egl_render_manager->vertex_buffers(vbuf_id);
      ldbg << "EGL : update vertex buffer " << *vertex_buffer << " , nv="<< vcount << " , id="<<vbuf_id<<std::endl;
      glvbos.set_number_of_vertices( vcount );
      GLfloat * vdata = (GLfloat*) glvbos.host_map_write_only(0);
      for(long i=0;i<vcount;i++)
      {
        const auto v = mesh->vertex(i);
        vdata[i*3+0] = v.x;
        vdata[i*3+1] = v.y;
        vdata[i*3+2] = v.z;
      }
      glvbos.host_unmap(0);
      
      int ebuf_id = egl_render_manager->element_buffer_id( *element_buffer );
      if( ebuf_id == -1 )
      {
        ldbg << "EGL : create element buffer " << *element_buffer <<std::endl;
        ebuf_id = egl_render_manager->create_element_buffer( *element_buffer , mesh->triangle_count() * 3 );
      }
      
      auto & ebuf = egl_render_manager->element_buffer(ebuf_id);
      ldbg << "EGL : update element buffer " << *element_buffer << " , triangles="<< tcount << " , id="<<ebuf_id<<std::endl;
      auto * elptr = ebuf.map_buffer_write_only();
      for(long i=0;i<tcount;i++)
      {
        const auto tc = mesh->triangle_connectivity(i);
        for(int j=0;j<3;j++) elptr[i*3+j] = tc[j];
      }
      ebuf.unmap_buffer();
    }

  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(egl_trimesh_to_buffer)
  {
    OperatorNodeFactory::instance()->register_factory( "egl_trimesh_to_buffer", make_compatible_operator< EGLTriMeshToBuffer > );
  }

}

