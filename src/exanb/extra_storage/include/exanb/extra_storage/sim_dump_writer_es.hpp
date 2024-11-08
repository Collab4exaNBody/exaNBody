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

#include <iostream>
#include <fstream>
#include <string>

#include <exanb/io/sim_dump_writer.h>
#include <exanb/extra_storage/dump_filter_dynamic_data_storage.h>

namespace exanb
{

	template<class GridT, class ESType, typename DumpFieldSet>
		class SimDumpWriteParticleES : public OperatorNode
	{
		ADD_SLOT( MPI_Comm    , mpi             , INPUT );
		ADD_SLOT( GridT       , grid     , INPUT );
		ADD_SLOT( Domain      , domain   , INPUT );
		ADD_SLOT( std::string , filename , INPUT );
		ADD_SLOT( long        , timestep      , INPUT , DocString{"Iteration number"} );
		ADD_SLOT( double      , physical_time , INPUT , DocString{"Physical time"} );
		ADD_SLOT( long        , compression_level , INPUT , 0 , DocString{"Zlib compression level"} );
		ADD_SLOT( GridExtraDynamicDataStorageT<ESType> , ges  , INPUT_OUTPUT , DocString{"ESType list"} );

		public:
		inline void execute () override final
		{
			auto optional_extra_data = ParticleDumpFilterWithExtraDataStorage<GridT, ESType, DumpFieldSet>{*ges,*grid};

			exanb::write_dump( *mpi, ldbg, *grid, *domain, *physical_time, *timestep, *filename, *compression_level, DumpFieldSet{} , optional_extra_data );
		}
	};
}

