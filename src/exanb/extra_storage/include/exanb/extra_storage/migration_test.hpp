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

#include <cassert>
#include <cstdint>
#include <tuple>
#include <exanb/extra_storage/extra_storage_info.hpp>

namespace exanb
{
	namespace migration_test
	{
		using namespace std;
    using InfoType = ExtraStorageInfo; 
    using UIntType = ExtraStorageInfo::UIntType;

		inline bool check_info_consistency(const InfoType* __restrict__ info_ptr, const UIntType info_size)
		{
			for (size_t p = 0 ; p < info_size ; p++)
			{
				auto [offset, size, id] = info_ptr[p];
				if(p == 0)
				{
					if(offset != 0) return false;
				}
				else
				{
					auto [last_offset, last_size, last_id] = info_ptr[p-1];
					if(offset != last_offset + last_size) return false;
				}
			}
			return true;
		}

		inline bool check_info_value(const InfoType* __restrict__ info_ptr, const UIntType info_size, UIntType value)
		{
			for (size_t p = 0 ; p < info_size ; p++)
			{
				if( info_ptr[p].size >= value) return false;
			}
			return true;
		}

		inline bool check_info_doublon(const InfoType* __restrict__ info_ptr, const UIntType info_size)
		{
			for (size_t p1 = 0 ; p1 < info_size ; p1++)
			{
				for (size_t p2 = p1 + 1 ; p2 < info_size ; p2++)
				{
					if ( info_ptr[p1] == info_ptr[p2] )
					{
						return false;
					}
				}
			}
			return true;
		}
	}
}
