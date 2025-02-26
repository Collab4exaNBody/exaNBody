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
#pragma once

#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/grid_cell_particles/grid_cell_values.h>
#include <exanb/core/grid_particle_field_accessor.h>
#include <onika/math/quaternion_operators.h>
#include <onika/cuda/cuda.h>

namespace exanb
{

  namespace GridCellValuesUtils
  {
    template<class T, class = void> struct FieldTypeSupportWeighting : std::false_type {};
    template<class T> struct FieldTypeSupportWeighting<T, decltype(void(sizeof(T{}*1.0))) > : std::true_type {}; 
    template<class T> static inline constexpr bool type_support_weighting_v = FieldTypeSupportWeighting<T>::value ;
  
    using StringList = std::vector<std::string>;
  
    ONIKA_HOST_DEVICE_FUNC static inline void localize_subcell( const Vec3d& r, double cell_size, double sub_cellsize, ssize_t subdiv, IJK& cell_loc, IJK& subcell_loc )
    {
      cell_loc = make_ijk( r / cell_size );
      Vec3d ro = r - (cell_loc*cell_size);
      subcell_loc = vclamp( make_ijk(ro / sub_cellsize) , 0 , subdiv-1 );
    }

    inline void subcell_neighbor( const IJK& cell_loc, const IJK& subcell_loc, ssize_t subdiv, IJK ninc, IJK& nbh_cell_loc, IJK& nbh_subcell_loc )
    {
      nbh_cell_loc = cell_loc;
      nbh_subcell_loc = subcell_loc + ninc;
      if(nbh_subcell_loc.i<0) { -- nbh_cell_loc.i; } else if(nbh_subcell_loc.i>=subdiv) { ++ nbh_cell_loc.i; }
      if(nbh_subcell_loc.j<0) { -- nbh_cell_loc.j; } else if(nbh_subcell_loc.j>=subdiv) { ++ nbh_cell_loc.j; }
      if(nbh_subcell_loc.k<0) { -- nbh_cell_loc.k; } else if(nbh_subcell_loc.k>=subdiv) { ++ nbh_cell_loc.k; }
      nbh_subcell_loc.i = ( nbh_subcell_loc.i + subdiv ) % subdiv;
      nbh_subcell_loc.j = ( nbh_subcell_loc.j + subdiv ) % subdiv;
      nbh_subcell_loc.k = ( nbh_subcell_loc.k + subdiv ) % subdiv;      
    }

    // @return how much of this particle contributes to region cell_box.
    // sum of contributions for all disjoint cell_box paving the domain is guaranteed to be 1.0
    inline double particle_weight(const Vec3d& r, double sp_size, const AABB& cell_box)
    {
      AABB contrib_box = { r - sp_size*0.5 , r + sp_size*0.5 };
      AABB sub_contrib_box = intersection( contrib_box , cell_box );
      double w = 0.0;
      if( ! is_nil(sub_contrib_box) ) { w = bounds_volume(sub_contrib_box) / (sp_size*sp_size*sp_size); }
      assert( w>=0. && w<=(1.0+1.e-11) );
      return w;
    }


  } // GridCellValuesUtils
  
} // ecanb

