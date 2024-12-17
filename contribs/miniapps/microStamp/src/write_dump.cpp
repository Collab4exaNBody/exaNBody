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

#include <onika/log.h>
#include <onika/math/basic_types_yaml.h>
#include <onika/math/basic_types_stream.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>

#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>

#include <iostream>
#include <fstream>
#include <string>

#include <exanb/io/sim_dump_writer.h>

namespace exaStamp
{
  using namespace exanb;

  template<
    class GridT,
    class = AssertGridHasFields< GridT, field::_rx,field::_ry,field::_rz, field::_vx,field::_vy,field::_vz, field::_id, field::_type >
    >
  class WriteDump : public OperatorNode
  {
    ADD_SLOT( MPI_Comm    , mpi             , INPUT );
    ADD_SLOT( GridT       , grid     , INPUT );
    ADD_SLOT( Domain      , domain   , INPUT );
    ADD_SLOT( std::string , filename , INPUT , std::string("output.dump") );
    ADD_SLOT( long        , timestep      , INPUT , DocString{"Iteration number"} );
    ADD_SLOT( double      , physical_time , INPUT , DocString{"Physical time"} );
    ADD_SLOT( long        , compression_level , INPUT , 6 , DocString{"Zlib compression level"} );
    ADD_SLOT( long        , max_part_size , INPUT , -1 , DocString{"Maximum file partition size. set -1 for system default value"} );

  public:
    inline void execute () override final
    {
      static constexpr FieldSet< field::_rx,field::_ry,field::_rz, field::_vx,field::_vy,field::_vz, field::_id, field::_type > dump_fields = {};
      static constexpr NullDumpOptionalFilter no_optional_data = {};
      
      size_t mps = MpiIO::DEFAULT_MAX_FILE_SIZE;
      if( *max_part_size > 0 ) mps = *max_part_size;
      exanb::write_dump( *mpi, ldbg, *grid, *domain, *physical_time, *timestep, *filename, *compression_level, dump_fields, no_optional_data, mps );
    }
  };

  template<class GridT> using WriteDumpTmpl = WriteDump<GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(write_dump)
  {
    OperatorNodeFactory::instance()->register_factory( "write_dump" , make_grid_variant_operator<WriteDumpTmpl> );
  }

}

