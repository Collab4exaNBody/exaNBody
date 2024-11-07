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
#include <onika/log.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/fields.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/quaternion_stream.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <onika/print_utils.h>
#include <exanb/core/print_particle.h>

#include <onika/soatl/field_tuple.h>

#include <vector>
#include <map>
#include <algorithm>
#include <sstream>

namespace exanb
{

  // ================== Thermodynamic state compute operator ======================

  template<typename GridT
    , class = AssertGridHasFields< GridT, field::_id >
    >
  class DebugParticleNode : public OperatorNode
  {
    using ParticleIds = std::vector<uint64_t>;

    ADD_SLOT( GridT       , grid      , INPUT, REQUIRED);
    ADD_SLOT( ParticleIds , ids       , INPUT, REQUIRED);
    ADD_SLOT( bool        , ghost     , INPUT, false );

  public:
  
    inline void execute () override final
    {
      ParticleIds ids = *(this->ids);
      std::sort( ids.begin(), ids.end() );

      auto cells = grid->cells();
      IJK dims = grid->dimension();
      
      std::map<uint64_t,std::vector<std::string> > dbg_items;

#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN( dims, i, loc )
        {
          const uint64_t* __restrict__ part_ids = cells[i][field::id];
          bool is_ghost_cell = grid->is_ghost_cell( loc );
          size_t n_part = cells[i].size();
          for(size_t j=0;j<n_part;j++)
          {
            if( ( ids.empty() || std::binary_search( ids.begin(), ids.end(), part_ids[j] ) ) && ( (*ghost) || !is_ghost_cell ) )
            {
              std::ostringstream oss;
              oss<< onika::default_stream_format;
              oss<<"---- PARTICLE "<<part_ids[j]<<" ";
              if(is_ghost_cell) { oss<<"GHOST"; }
              oss<<"----"<<std::endl<<"cell = " << loc <<std::endl;
              print_particle( oss , cells[i][j] );
              oss<<"------------------------------------------";
#             pragma omp critical
              {
                dbg_items[ part_ids[j] ].push_back( oss.str() );
              }
            }
          }
        }
        GRID_OMP_FOR_END
      }

      for( const auto& x : dbg_items ) for( const auto& y : x.second )
      {
        lout << y << std::endl;
      }
      
    }

  };
  
  template<class GridT> using DebugParticleNodeTmpl = DebugParticleNode<GridT>;
  
  // === register factories ===
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "debug_particle", make_grid_variant_operator<DebugParticleNodeTmpl> );
  }

}

