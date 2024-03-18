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

#include <include/exanb/extra_storage/move_particles_es.hpp>

namespace exanb
{
	template <typename GridT> using MovePaticlesWithDoubleTmpl = MovePaticlesWithES<GridT, double>;
	template <typename GridT> using MovePaticlesWithIntTmpl = MovePaticlesWithES<GridT, int>;
	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "move_particles_double", make_grid_variant_operator< MovePaticlesWithDoubleTmpl> );
		OperatorNodeFactory::instance()->register_factory( "move_particles_int", make_grid_variant_operator< MovePaticlesWithIntTmpl> );
	}
}

