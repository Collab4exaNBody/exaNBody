#include <exanb/core/units.h>

#include <cstring>
#include <string>
#include <vector>
#include <utility>
#include <cassert>
#include <sstream>
#include <cstdlib>

#include "tinyexpr.h"

namespace exanb
{

  namespace units
  {

    Quantity make_quantity( double value , const std::string& units_and_powers )
    {
      std::string p_unities = units_and_powers;
      // Take in input unities
      // Split unities, and take the power associated
      // Example :
      // input s="ang^6/kcal/J/T^7/kg^30"
      // output unities_list = ({ang,6}, {kcal,-1}, {J,-1}, {T,-7}, {kg,-30})
      size_t pos = 0;
      double power = 1.0;//first unit positive
      std::vector< std::pair<std::string, double> > unities_list;
      while ((pos = p_unities.find_first_of("/*.")) != std::string::npos)
      {
        unities_list.push_back(std::make_pair(p_unities.substr(0, pos), power));
        if((p_unities.compare(pos, 1, "/"))==0) power = -1.0;
        else power = 1.0;
        p_unities.erase(0, pos + 1);
      }
      unities_list.push_back( std::make_pair(p_unities.substr(0, pos), power) );//to take into account the last unit

      double power_value = 0.0;
      for(auto& unity: unities_list)
      {
        if((pos=unity.first.find("^")) != std::string::npos)
	      {
	        //allows conversion between char and int
	        std::stringstream ss(unity.first.substr(pos+1,unity.first.size()-pos));
	        ss >> power_value;
	        unity.second *= power_value;

	        unity.first.erase(pos,unity.first.size());
	        power_value=0;
	      }
      }
      
      Quantity q = { value , SI };
      for(const auto& pair : unities_list)
      {
        //Protection : if the unity doesn't exist, exit the code
        q = q * ( unit_from_symbol(pair.first) ^ pair.second );
      }
      return q;    
    }
    
    Quantity quantity_from_string(const std::string& s,bool& conversion_done)
    {
      conversion_done = false;

      Quantity q = { 0.0 , SI };    
      std::istringstream iss( s );
      std::string unit_str;
      std::string value_str;
      
      iss >> value_str ; 
      
      // 1. try to read first item as a numerical value
      char* endptr = nullptr;
      q.m_value = std::strtod( value_str.c_str() , &endptr );    
      size_t value_len = endptr - value_str.c_str();
      
      // 2. try to read first item as an arithmetic expression (if it contains other symbols than those in numbers AND does not contain space separator after the first number )
      // FIXME: find another format to disambiguate expressions vs number with quantity, i.e. with surrounding character, like $3.0*3+2$
      if( value_len != value_str.length() )
      {
        int te_err = 0;
        q.m_value = te_interp(value_str.c_str(),&te_err); // expression without units, will give unitless value
        if( te_err!=0 )
        {
          conversion_done = false;
          return q;
        }
      }
      
      // if a separator is detected, we know there is 
      if( s.find_first_of(" \t\n") != std::string::npos )
      {
        iss >> unit_str;
        q = make_quantity( q.m_value , unit_str );
      }
      conversion_done = true;
      return q;
    }

    Quantity quantity_from_string(const std::string& s)
    {
      bool dont_care = false;
      return quantity_from_string(s,dont_care);
    }

  } // end of namespace unit
  
} // end of namespace exanb


