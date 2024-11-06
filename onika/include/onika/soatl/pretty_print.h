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

#include <onika/soatl/field_id.h>

#include <tuple>

namespace onika { namespace soatl {

template<typename StreamT, typename ArrayT, typename... ids>
static inline StreamT& pretty_print_arrays(StreamT& out, const ArrayT& arrays, const FieldId<ids>& ... )
{
  out << "alignment="<< arrays.alignment() << "\n";
  out << "chunksize="<< arrays.chunksize() << "\n";
  out << "size="<< arrays.size() << "\n";
  out << "datasize="<< arrays.storage_size() << "\n";
  out << "contains "<< sizeof...(ids) << " fields :\n";
  TEMPLATE_LIST_BEGIN
    out<<"  "<< soatl::FieldId<ids>::name() << " : type=" << typeid(typename soatl::FieldId<ids>::value_type).name() <<", addr="<<(void*) arrays[FieldId<ids>()] <<"\n"
  TEMPLATE_LIST_END  
  return out;
}
  
template<typename StreamT, typename ArrayT, typename... ids>
static inline StreamT& pretty_print_arrays(StreamT& out, const ArrayT& arrays, const std::tuple< FieldId<ids> ... > & )
{
  return pretty_print_arrays( out, arrays, FieldId<ids>() ... );
}

template<typename StreamT, typename ArrayT>
static inline StreamT& pretty_print_arrays(StreamT& out, const ArrayT& arrays)
{
  return pretty_print_arrays( out, arrays, typename ArrayT::FieldIdsTuple () );
}
  
} // namespace soatl
}

