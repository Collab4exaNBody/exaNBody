#include <exanb/core/quantity.h>
#include <exanb/core/unityConverterHelper.h>
#include <cassert>
#include <sstream>
#include <cstdlib>

#include "tinyexpr.h"

namespace exanb
{

  double Quantity::convert( const UnitSystem& us ) const
  {
    return UnityConverterHelper::convert( m_value, m_unit , 
                          us.m_length ,
                          us.m_mass ,
                          us.m_time ,
                          us.m_electric_current ,
                          us.m_temperature ,
                          us.m_amount_of_substance ,
                          us.m_luminous_intensity ,
                          us.m_angle );
  }

  const Quantity& Quantity::max( const Quantity& rhs )
  {
    if( convert() < rhs.convert() ) { return rhs; }
    else { return *this; }
  }

  Quantity Quantity::operator + ( const Quantity& rhs )
  {
    assert( m_unit == rhs.m_unit );
    return Quantity{ m_value + rhs.m_value , m_unit };
  }

  Quantity make_quantity(double x, const std::string& u)
  {
    Quantity q;
    q.m_value = x;
    UnityConverterHelper::parse_unities_descriptor( u, q.m_unit );
    return q;
  }

  Quantity quantity_from_string(const std::string& s)
  {
    bool dont_care;
    return quantity_from_string(s,dont_care);
  }
  
  Quantity quantity_from_string(const std::string& s,bool& conversion_done)
  {
    conversion_done = false;
    Quantity q;
    std::istringstream iss( s );
    std::string unit_str;
    std::string value_str;
    
    iss >> value_str ; 
    
    // 1. try to read first item as a numerical value
    char* endptr = nullptr;
    q.m_value = std::strtod( value_str.c_str() , &endptr );    
    size_t value_len = endptr - value_str.c_str();
    
    // 2. try to read first item as an arithmetic expression
    if( value_len != value_str.length() )
    {
      int te_err = 0;
      q.m_value = te_interp(value_str.c_str(),&te_err);
      if( te_err!=0 )
      {
        conversion_done = false;
        return q;
      }
    }
    
    if( s.find_first_of(" \t\n") != std::string::npos )
    {
      iss >> unit_str;
      UnityConverterHelper::parse_unities_descriptor( unit_str, q.m_unit );
    }
    else
    {
      q.m_unit.assign( 1 , exanb::UnityWithPower{ exanb::EnumUnities::no_unity , 1 } );
    }
    conversion_done = true;
    return q;
  }

}

