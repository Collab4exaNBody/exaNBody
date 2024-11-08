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
#include <onika/scg/operator_slot_base.h>
#include <onika/cpp_utils.h>
#include <onika/log.h>
#include <cstdint>
#include <cstring>

namespace onika { namespace scg
{
    
  // === register factories ===  
  CONSTRUCTOR_ATTRIB void _register_integer_conversions()
  {
    ldbg << "register signed/unsigned integer conversions" << std::endl;
    OperatorSlotBase::register_type_conversion_force_cast<int64_t,uint64_t>();
    OperatorSlotBase::register_type_conversion_force_cast<uint64_t,int64_t>();
    OperatorSlotBase::register_type_conversion_force_cast<int32_t,uint32_t>();
    OperatorSlotBase::register_type_conversion_force_cast<uint32_t,int32_t>();

    int64_t x = -( (1ll<<30) + 1 );
    int64_t* px64 = &x;
    int32_t* px32 = nullptr;
    assert( sizeof(px32) == sizeof(px64) );
    std::memcpy( &px32 , &px64 , sizeof(px64) ); // according to c++ norm, this is ok while a reinterpret_cast is not
    bool endianness_ok = ( *px32 == *px64 );

    if( endianness_ok )
    {
      ldbg << "register 64 bits to 32 bits integer conversions" << std::endl;
      OperatorSlotBase::register_type_conversion_force_cast<int64_t ,int32_t >();
      OperatorSlotBase::register_type_conversion_force_cast<int64_t ,uint32_t>();
      OperatorSlotBase::register_type_conversion_force_cast<uint64_t,int32_t >();
      OperatorSlotBase::register_type_conversion_force_cast<uint64_t,uint32_t>();
    }
    else
    {
      lerr << "Warning: no 64 bits to 32 bits integer conversion" << std::endl;
    }
  }
  
} }

