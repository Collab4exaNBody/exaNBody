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
#include <onika/math/basic_types_yaml.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid.h>
#include <onika/math/basic_types_stream.h>
#include <onika/log.h>
#include <exanb/core/domain.h>
#include <exanb/core/simple_block_rcb.h>
#include <exanb/io/grid_memory_compact.h>

#include <iostream>
#include <fstream>
#include <string>

#include <exanb/io/sim_dump_reader.h>
#include <exanb/extra_storage/reader_dynamic_data_storage.hpp>
#include <exanb/extra_storage/dump_filter_dynamic_data_storage.h>

namespace exanb
{
  template<class GridT , class ESType, typename DumpFieldSet>
    class SimDumpReadParticlesES : public OperatorNode
  {
    ADD_SLOT( MPI_Comm    , mpi             , INPUT );
    ADD_SLOT( GridT       , grid     , INPUT_OUTPUT );
    ADD_SLOT( Domain      , domain   , INPUT_OUTPUT );
    ADD_SLOT( std::string , filename , INPUT );
    ADD_SLOT( long        , timestep      , INPUT , DocString{"Iteration number"} );
    ADD_SLOT( double      , physical_time , INPUT , DocString{"Physical time"} );

    ADD_SLOT( double      , scale_cell_size , INPUT , OPTIONAL );
    ADD_SLOT( bool        , enable_extra_data , INPUT , true );
    ADD_SLOT( std::vector<bool> , periodicity     , INPUT ,OPTIONAL , DocString{"if set, overrides domain's periodicity stored in file with this value"}  );
    ADD_SLOT( bool        , expandable      , INPUT ,OPTIONAL , DocString{"if set, override domain expandability stored in file"} );
    ADD_SLOT( AABB        , bounds          , INPUT ,OPTIONAL , DocString{"if set, override domain's bounds, filtering out particle outside of overriden bounds"} );
    ADD_SLOT( bool        , shrink_to_fit   , INPUT ,OPTIONAL , DocString{"if set to true and bounds was wpecified, try to reduce domain's grid size to the minimum size enclosing fixed bounds"} );
    ADD_SLOT( GridExtraDynamicDataStorageT<ESType> , ges  , INPUT_OUTPUT , DocString{"Interaction list"} );

    public:
    inline void execute () override final
    {
      ParticleDumpFilterWithExtraDataStorage<GridT, ESType, DumpFieldSet> dump_filter = {*ges,*grid};

      if( scale_cell_size.has_value() )
      {
        dump_filter.scale_cell_size = *scale_cell_size;
        ldbg << "force cell size scaling to "<<dump_filter.scale_cell_size<<std::endl;
      }


      if( periodicity.has_value() )
      {
        dump_filter.override_periodicity = true;
        if( periodicity->size() >= 1 ) dump_filter.periodic_x = periodicity->at(0);
        if( periodicity->size() >= 2 ) dump_filter.periodic_y = periodicity->at(1);
        if( periodicity->size() >= 3 ) dump_filter.periodic_z = periodicity->at(2);
        ldbg << "force periodicity to ("<<std::boolalpha<<dump_filter.periodic_x<<","<<dump_filter.periodic_y<<","<<dump_filter.periodic_z<<")" <<std::endl;
      }
      if( expandable.has_value() )
      {
        ldbg << "force expandability to "<<std::boolalpha<< *expandable << std::endl;
        dump_filter.override_expandable = true;
        dump_filter.expandable = *expandable;
      }
      if( bounds.has_value() )
      {
        ldbg << "force domain bounds to "<< *bounds << std::endl;
        dump_filter.override_domain_bounds = true;
        dump_filter.domain_bounds = *bounds;
        if( shrink_to_fit.has_value() )
        {
          dump_filter.shrink_to_fit = *shrink_to_fit;
        }
      }

      dump_filter.enable_extra_data = *enable_extra_data;
      exanb::read_dump( *mpi, ldbg, *grid, *domain, *physical_time, *timestep, *filename, DumpFieldSet{} , dump_filter );
      exanb::grid_memory_compact(*grid);
    }
  };
}
