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
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>

#include <memory>
#include <exanb/extra_storage/migration_test.hpp>
#include <exanb/extra_storage/dynamic_data_storage.hpp>

namespace exanb
{
	// GridExtraDataStorage -> GridES
	template<typename GridT, typename GridES
		, class = AssertGridHasFields< GridT >
		>
		class CheckInfoConsistency : public OperatorNode
		{
			ADD_SLOT( GridT       , grid              , INPUT_OUTPUT , REQUIRED );
			ADD_SLOT( GridES      , ges  , INPUT , REQUIRED,  DocString{"Grid of Extra Data Storage (per cell)"} );

			public:

			inline std::string documentation() const override final
			{
				return R"EOF(
					"This opertor checks if for each particle information the offset and size are correct"
				        )EOF";
			}

			inline void execute () override final
			{
				if( grid->number_of_cells() == 0 ) { return; }
				auto & ces = ges->m_data; // cell extra storage

				for(size_t current_cell = 0 ; current_cell < ces.size() ; current_cell++)
				{
					auto storage = ces[current_cell];
					size_t n_particles_stored = storage.number_of_particles();
					auto* info_ptr = storage.m_info.data();
					[[maybe_unused]] bool is_okay = migration_test::check_info_consistency( info_ptr, n_particles_stored);
					assert(is_okay && "CheckConsistency");
				}
			}
		};
}

