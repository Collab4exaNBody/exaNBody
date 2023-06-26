#pragma once

#include <exanb/core/enum_unities.h>
#include <string>

namespace exanb
{
  
  struct Quantity
  {    
    double convert( const UnitSystem& us = UnitSystem() ) const;
    const Quantity& max( const Quantity& rhs );
    Quantity operator + ( const Quantity& rhs );
  
    double m_value = 0.0;
    UnitiesDescriptor m_unit;
  };

  inline Quantity make_simple_quantity(double x, EnumUnities u)
  {
    return Quantity { x, { { u , 1 } } };
  }
  
  Quantity make_quantity(double x, const std::string& u);
  Quantity quantity_from_string(const std::string& v_u);
  Quantity quantity_from_string(const std::string& s,bool& conversion_done);
}

