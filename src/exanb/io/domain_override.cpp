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
#include <exanb/core/log.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/domain.h>
#include <exanb/core/string_utils.h>

#include <iostream>
#include <string>

namespace exanb
{

  class DomainOverride : public OperatorNode
  {
    using BoolVector = std::vector<bool>;
    using StringVector = std::vector<std::string>;

    ADD_SLOT( Domain       , domain     , INPUT_OUTPUT );
    ADD_SLOT( double       , cell_size  , INPUT , OPTIONAL );
    ADD_SLOT( AABB         , bounds     , INPUT , OPTIONAL );
    ADD_SLOT( IJK          , grid_dims  , INPUT , OPTIONAL );
    ADD_SLOT( bool         , expandable , INPUT , OPTIONAL );
    ADD_SLOT( BoolVector   , periodic   , INPUT , OPTIONAL );
    ADD_SLOT( StringVector , mirror     , INPUT ,OPTIONAL , DocString{"if set, overrides domain's boundary mirror flags in file with provided values"}  );

  public:
    inline void execute() override final
    {
      if( cell_size.has_value() )
      {
        lout << "override domain's cell size to "<< *cell_size << std::endl;
        domain->set_cell_size( *cell_size );
      }
      
      if( bounds.has_value() )
      {
        lout << "override domain's bounds to "<< *bounds << std::endl;
        domain->set_bounds( *bounds );
      }

      if( grid_dims.has_value() )
      {
        lout << "override domain's grid_dims to "<< *grid_dims << std::endl;
        domain->set_grid_dimension( *grid_dims );
      }

      if( periodic.has_value() )
      {
        auto p = *periodic;
        p.resize(3,false);
        lout << "override domain's periodicity to "<<p[0]<<" "<<p[1]<<" "<<p[2]<< std::endl;
        domain->set_periodic_boundary( p[0], p[1], p[2] );
      }

      if( expandable.has_value() )
      {
        lout << "override domain's expandable to "<< *expandable << std::endl;
        domain->set_expandable( *expandable );
      }
      if( mirror.has_value() )
      {
        domain->set_mirror_x_min(false); domain->set_mirror_x_max(false); 
        domain->set_mirror_y_min(false); domain->set_mirror_y_max(false); 
        domain->set_mirror_z_min(false); domain->set_mirror_z_max(false); 
        lout << "override domain's boundary mirror flags to [";
        for(auto m : *mirror)
        {
          lout << " "<<m;
          if( exanb::str_tolower(m) == "x-" ) { domain->set_mirror_x_min(true); }
          if( exanb::str_tolower(m) == "x+" ) { domain->set_mirror_x_max(true); }
          if( exanb::str_tolower(m) == "x" )  { domain->set_mirror_x_min(true); domain->set_mirror_x_max(true); }
          if( exanb::str_tolower(m) == "y-" ) { domain->set_mirror_y_min(true); }
          if( exanb::str_tolower(m) == "y+" ) { domain->set_mirror_y_max(true); }
          if( exanb::str_tolower(m) == "y" )  { domain->set_mirror_y_min(true); domain->set_mirror_y_max(true); }
          if( exanb::str_tolower(m) == "z-" ) { domain->set_mirror_z_min(true); }
          if( exanb::str_tolower(m) == "z+" ) { domain->set_mirror_z_max(true); }
          if( exanb::str_tolower(m) == "z" )  { domain->set_mirror_z_min(true); domain->set_mirror_z_max(true); }
        }
        lout << " ]"<<std::endl;
      }

      ldbg<<"domain new values : "<< *domain << std::endl;
      //check_domain( *domain );
    }

  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "domain_override", make_simple_operator<DomainOverride> );
  }

}

