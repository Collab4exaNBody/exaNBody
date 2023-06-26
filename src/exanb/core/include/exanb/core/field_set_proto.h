#pragma once

#include <exanb/fields.h>
#include <onika/soatl/field_id.h>

namespace exanb
{
  template<typename... field_ids> using FieldSet = onika::soatl::FieldIds< field_ids... > ;
  template<typename... field_sets> struct FieldSets {};
}

