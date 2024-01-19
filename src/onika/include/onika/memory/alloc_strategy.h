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

namespace onika { namespace memory
{
	struct ChunkIncrementalAllocationStrategy
	{
		static inline size_t update_capacity(size_t s, size_t capacity, size_t chunksize)
		{
			size_t newCapacity = 0;
			if( s>capacity || ( (s+2*chunksize) <= capacity ) || s==0 )
			{
				newCapacity = ( (s+chunksize-1)/chunksize ) * chunksize ;
			}
			else
			{
				newCapacity = capacity;
			}
			assert( newCapacity >= s );
			assert( (newCapacity % chunksize) == 0 );
			return newCapacity;
		}
	};

	struct ChunkLogAllocationStrategy
	{
		static inline size_t update_capacity(size_t s, size_t capacity, size_t chunksize)
		{
			size_t newCapacity = 0;
			if( s>capacity )
			{
        if( capacity <= s/2 || capacity==0 ) newCapacity = s;
        else newCapacity = capacity * 2;
			}
			else if( ( s <= capacity/2 ) || s==0 )
			{
				newCapacity = s ;
			}
			else
			{
				newCapacity = capacity;
			}
			newCapacity = ( (newCapacity+chunksize-1)/chunksize ) * chunksize;			
			assert( newCapacity >= s );
			assert( (newCapacity % chunksize) == 0 );
			return newCapacity;
		}
	};

	using DefaultAllocationStrategy = ChunkLogAllocationStrategy;

}

}

