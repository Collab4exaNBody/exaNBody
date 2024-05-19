
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

#include <exanb/core/grid.h>
#include <memory>
#include <vector>
#include <cstddef>

namespace exanb
{
	struct ExtraStorageInfo
	{
		using UIntType = uint64_t;
		//using UIntType = uint32_t;
		UIntType offset; ///< start / offset 
		UIntType size;  ///< number of items
		uint64_t pid;    ///< Particle Id

		bool operator == (const ExtraStorageInfo& in) const
		{
			return (this->offset == in.offset ) && (this->size == in.size) && (this->pid == in.pid);
		}
	};
}
