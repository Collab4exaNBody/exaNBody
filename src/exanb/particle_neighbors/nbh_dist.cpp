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
#include <exanb/core/domain.h>

#include <memory>

namespace exanb
{
  

  struct NeighborDistanceNode : public OperatorNode
  {
    ADD_SLOT(double , rcut_max          , INPUT , 0.0 , DocString{"maximum search distance for the neighborghood, in physical space"} );
    ADD_SLOT(double , rcut_inc          , INPUT , 0.0 , DocString{"value added to the search distance to update neighbor list less frequently. in physical space"} );
    ADD_SLOT(double , ghost_dist_max    , INPUT_OUTPUT , 0.0 , DocString{"maximum distance needed for ghost particles out of sub domain, in lab (physical) space."} );
    ADD_SLOT(double , bond_max_dist     , INPUT , 0.0 , DocString{"molecule bond max distance, in physical space"} );
    ADD_SLOT(double , bond_max_stretch  , INPUT , 0.0 , DocString{"fraction of bond_max_dist."} );
    ADD_SLOT(Domain , domain            , INPUT , REQUIRED );
    ADD_SLOT(bool   , verbose           , INPUT , false );
    ADD_SLOT(double , nbh_dist_lab      , INPUT_OUTPUT , DocString{"neighborhood distance, in lab space"} );
    ADD_SLOT(double , nbh_dist          , INPUT_OUTPUT , DocString{"neighborhood distance, in grid space"} );
    ADD_SLOT(double , ghost_dist        , INPUT_OUTPUT , DocString{"thickness of ghost particle layer, in grid space"} );
    ADD_SLOT(double , max_displ         , INPUT_OUTPUT , DocString{"move threshold, in grid space, that must trigger a neighbor list update (verlet method)"} );

    inline void execute () override final
    {
      *nbh_dist_lab = *rcut_max + *rcut_inc;
      *max_displ = (*rcut_inc) / 2.0; 
      ldbg << "rcut_max+rcut_inc = " << *nbh_dist_lab << std::endl;

      const double min_scale = domain->xform_min_scale();
      const double max_scale = domain->xform_max_scale();

      if(*verbose)
      {
        lout << "========= Neighborhood ==========" << std::endl;
        if( ! domain->xform_is_identity() )
        {
          lout << "transform      = "<< domain->xform() << std::endl;
          lout << "scale          = "<< min_scale << " / "<< max_scale << std::endl;
        }
      }
      
      *nbh_dist = (*nbh_dist_lab) / min_scale;  // lab space neighborhood distance => grid space distance
      *max_displ /= max_scale; // lab space scaling => grid space scaling
      const double bsdist = (*bond_max_dist) * ( 1. + (*bond_max_stretch) );
      const double bond_dist = ( bsdist + (*rcut_inc) ) / min_scale;
      
      *ghost_dist_max = std::max( *ghost_dist_max , *rcut_max );
      const double ghost_dist_max_grid = ( (*ghost_dist_max) + (*rcut_inc) ) / min_scale;
      *ghost_dist = std::max( bond_dist , ghost_dist_max_grid );  // ghost distance needed, in grid space
      
      if(*verbose)
      {
        lout << "rcut_max         = "<< *rcut_max << std::endl;
        lout << "ghost_dist_max   = "<< *ghost_dist_max << std::endl;
        lout << "rcut_inc         = "<< *rcut_inc << std::endl;
        lout << "nbh_dist_lab     = "<< *nbh_dist_lab << std::endl;
        lout << "nbh_dist         = "<< *nbh_dist << std::endl;
        lout << "max_displ        = "<< *max_displ << std::endl;
        lout << "bond_max_dist    = "<< *bond_max_dist << std::endl;
        lout << "bond_max_stretch = "<< *bond_max_stretch << std::endl;
        lout << "ghost_dist       = "<< *ghost_dist << std::endl;
        lout << "=================================" << std::endl << std::endl;
      }
    }

  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory("nbh_dist" , make_simple_operator< NeighborDistanceNode > );
  }

}

