#pragma once

#include <exanb/core/quantity.h>
#include <exanb/core/unityConverterHelper.h>
#include <iostream>

namespace exanb
{
  // pretty printing
  inline std::ostream& operator << (std::ostream& out, const Quantity& q)
  {
    out << q.m_value << ' ' << UnityConverterHelper::unities_descriptor_to_string(q.m_unit) ;
    return out;
  }
  
}

