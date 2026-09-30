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
#include <exanb/core/domain.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/grid.h>

#include <exanb/mesh/triangle_mesh.h>
#include <exanb/mesh/triangle_math.h>
#include <onika/math/basic_types.h>

namespace exanb
{
  
  class TriangleMeshInit : public OperatorNode
  {
    ADD_SLOT( TriangleMesh , mesh    , INPUT_OUTPUT , OPTIONAL , DocString{"placeholder to initialize a mesh from user input"} );
    ADD_SLOT( long         , verbosity , INPUT , 0 );

  public:
    inline void execute() override final
    {
      ldbg << "mesh init : mesh has_value = "<<mesh.has_value()<<std::endl;
      if( *verbosity && mesh.has_value() )
      {
        if( *verbosity >= 2 ) mesh->to_stream( lout );
        else lout << "mesh has "<< mesh->vertex_count() << " vertices , "<<mesh->triangle_count() << " triangles"<<std::endl;
      }
    }
  };

   // === register factories ===  
  ONIKA_AUTORUN_INIT(trimesh_init)
  {
    OperatorNodeFactory::instance()->register_factory("trimesh_init",make_simple_operator< TriangleMeshInit >);
  }

}

