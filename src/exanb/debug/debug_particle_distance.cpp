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
#include <exanb/core/domain.h>
#include <onika/log.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/fields.h>
#include <onika/math/basic_types_stream.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <onika/print_utils.h>
#include <exanb/core/print_particle.h>

#include <onika/soatl/field_tuple.h>

#include <vector>
#include <map>

namespace exanb
{

  // ================== Thermodynamic state compute operator ======================

  template<typename GridT
    , class = AssertGridHasFields< GridT, field::_id >
    >
  class DebugParticleDistance : public OperatorNode
  {
    ADD_SLOT( GridT    , grid , INPUT, REQUIRED );
    ADD_SLOT( long     , id1  , INPUT, REQUIRED );
    ADD_SLOT( long     , id2  , INPUT, REQUIRED );
    ADD_SLOT( double   , dist_max  , INPUT, -1.0 );
    ADD_SLOT( Domain   , domain , INPUT , REQUIRED );

  public:
  
    inline void execute() override final
    {
      const uint64_t id_a = *id1;
      const uint64_t id_b = *id2;
      const double dmax = *dist_max;
      
      auto cells = grid->cells();
      IJK dims = grid->dimension();
      // ssize_t gl = grid->ghost_layers();

      const Mat3d mat = domain->xform();
      lout << "using XForm " << mat << std::endl;

      std::vector< std::pair<uint64_t,uint64_t> > a_instances;
      std::vector< std::pair<uint64_t,uint64_t> > b_instances;

#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN( dims, i, loc )
        {
          const uint64_t* __restrict__ part_ids = cells[i][field::id];
          size_t n_part = cells[i].size();
          for(size_t j=0;j<n_part;j++)
          {
            if( part_ids[j] == id_a )
            {
#             pragma omp critical(DebugParticleDistance_add_a)
              a_instances.push_back( {i,j} );
            }
            else if( part_ids[j] == id_b )
            {
#             pragma omp critical(DebugParticleDistance_add_b)
              b_instances.push_back( {i,j} );
            }
          }
        }
        GRID_OMP_FOR_END
      }

      for(const auto& a : a_instances)
      {
        bool a_ghost = grid->is_ghost_cell( a.first );
        Vec3d ra = { cells[a.first][field::rx][a.second] , cells[a.first][field::ry][a.second] , cells[a.first][field::rz][a.second] };
        Vec3d proj_ra = mat * ra;
        for(const auto& b : b_instances)
        {
          bool b_ghost = grid->is_ghost_cell( b.first );
          Vec3d rb = { cells[b.first][field::rx][b.second] , cells[b.first][field::ry][b.second] , cells[b.first][field::rz][b.second] };
          Vec3d proj_rb = mat * rb;
          Vec3d dr = rb-ra;
          double d = norm(dr);
          Vec3d proj_dr = mat * dr;
          double proj_d = norm(proj_dr);

          if( d<=dmax || dmax<0.0 )
          {
            lout << "#"<<id_a; //<<" i="<<a.first<<"."<<a.second;
            if( a_ghost ) lout << ".G";
            lout << " <-> #"<<id_b; //<<", i="<<b.first<<"."<<b.second;
            if( b_ghost ) lout << ".G";
            lout << std::endl;
            lout <<"\tra="<<ra<<" proj_ra="<<proj_ra<< std::endl;
            lout <<"\trb="<<rb<<" proj_rb="<<proj_rb<< std::endl;
            lout <<"\tdr="<<dr<<" d="<<d<< " d2="<<d*d <<std::endl;
            lout <<"\tproj_dr="<<proj_dr<<" proj_d="<<proj_d << " proj_d2="<< proj_d*proj_d <<std::endl;
          }
        }
      }
      
    }

  };
  
  template<class GridT> using DebugParticleDistanceTmpl = DebugParticleDistance<GridT>;
  
  // === register factories ===
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "debug_particle_distance", make_grid_variant_operator<DebugParticleDistanceTmpl> );
  }

}

