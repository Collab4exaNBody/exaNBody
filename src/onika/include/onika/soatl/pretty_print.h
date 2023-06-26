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

