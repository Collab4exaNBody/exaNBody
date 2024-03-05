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

#include <cstdlib>

namespace exanb
{

    struct DefaultNeighborFilterFunc
    {
      inline bool operator () (double d2, double rcut2,size_t cell_a, size_t p_a, size_t cell_b, size_t p_b) const
      {
        assert( cell_a!=cell_b || p_a!=p_b );
        return d2 > 0.0 && d2 <= rcut2;
      }
    };

    template<class GridT>
    struct NeighborFilterHalfSymGhost
    {
      const GridT& m_grid;
      bool m_half_symmetric = false;
      bool m_skip_ghost = false;
      inline bool operator () (double d2, double rcut2, size_t cell_a, size_t p_a, size_t cell_b, size_t p_b) const
      {
        assert( cell_a!=cell_b || p_a!=p_b );
        if( d2 > 0.0 && d2 <= rcut2 )
        {
          if( m_half_symmetric && ( cell_a<cell_b || ( cell_a==cell_b && p_a<p_b ) ) ) return false;
          if( m_skip_ghost && m_grid.is_ghost_cell(cell_b) ) return false;
          return true;
        }
        else return false;
      }
    };

}

