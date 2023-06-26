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

