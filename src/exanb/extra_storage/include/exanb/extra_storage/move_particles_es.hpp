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
#include <exanb/core/operator.h>
#include <exanb/core/domain.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid.h>
#include <exanb/core/check_particles_inside_cell.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <onika/log.h>
#include <exanb/core/thread.h>

#include <vector>

#include <exanb/grid_cell_particles/move_particles_across_cells.h>
#include <exanb/extra_storage/dynamic_data_storage.hpp>
#include <exanb/extra_storage/migration_buffer.hpp>
#include <exanb/extra_storage/migration_helper.hpp>

namespace exanb
{
	template<class GridT, class ESType>
		class MovePaticlesWithES : public OperatorNode
	{ 
		using ParticleT = typename exanb::MoveParticlesHelper<GridT>::ParticleT;
		using ParticleVector = typename exanb::MoveParticlesHelper<GridT>::ParticleVector;
		using MovePaticlesScratch = typename exanb::MoveParticlesHelper<GridT>::MovePaticlesScratch;

		ADD_SLOT( Domain         , domain , INPUT );
		ADD_SLOT( GridT          , grid , INPUT_OUTPUT );
		ADD_SLOT( ParticleVector , otb_particles , OUTPUT );
		ADD_SLOT( GridExtraDynamicDataStorageT<ESType> , ges   , INPUT_OUTPUT , REQUIRED, DocString{"ES list"} );
		ADD_SLOT( ExtraDynamicDataStorageCellMoveBufferT < ESType > , bes , INPUT_OUTPUT, ExtraDynamicDataStorageCellMoveBufferT<ESType> {} , DocString{"interaction data of particles moving outside the box"} );

		ADD_SLOT( MovePaticlesScratch, move_particles_scratch , PRIVATE );

		public:

		inline std::string documentation() const override final
		{
			return R"EOF(
        				This operator moves particles and extra data storage (es) across cells.
				        )EOF";
		}

		inline void execute () override final
		{
			bes->clear();
			ExtraDynamicStorageDataGridMoveBufferT <ESType> interaction_opt_buffer = { ges->m_data , *bes };
			exanb::move_particles_across_cells( ldbg, *domain, *grid, *otb_particles, *move_particles_scratch, interaction_opt_buffer );
			assert( interaction_opt_buffer.check() );
			assert( bes->check() );
		}    
	};
}

