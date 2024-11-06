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
#include <onika/physics/units.h>

#include <cstring>
#include <string>
#include <vector>
#include <utility>
#include <cassert>
#include <sstream>
#include <cstdlib>
#include <iostream>

#include "tinyexpr.h"

namespace onika
{

  namespace physics
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
      bool inverse_power = false;
      if( p_unities.find("1/")==0 ) 
      {
        p_unities = p_unities.substr(2);
        inverse_power = true;
      }
      
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
        UnitDefinition u = unit_from_symbol(pair.first);
        Quantity mq = u ^ pair.second;
        //std::cout<<"Unit^Power: "<<std::defaultfloat<<pair.first<<" ^ "<<pair.second<<" (unit="<<u.m_name<<") => "<< mq <<std::endl;
        //std::cout<<q<<" * "<<mq;
        q = q * mq;
        //std::cout<<" = "<<q<<std::endl;
      }
      //std::cout<<std::endl;
      if( inverse_power )
      {
        const double v = q.m_value;
        q = q ^ -1.0;
        q.m_value = v;
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

    std::ostream& units_power_to_stream (std::ostream& out, const Quantity& q)
    {
      bool bad_units = false;
      for(int i=0;i<NUMBER_OF_UNIT_CLASSES;i++)
      {
        if( q.m_system.m_units[i].m_class != i ) bad_units = true;
      }
      if( bad_units )
      {
        out << "<bad-unit>";
      }
      else
      {
        bool unit_found = false;
        for(int i=0;i<NUMBER_OF_UNIT_CLASSES;i++)
        {
          if( q.m_unit_powers.m_powers[i] != 0.0 )
          {
            out << ( unit_found ? "." : "" ) << q.m_system.m_units[i].m_short_name << '^' << q.m_unit_powers.m_powers[i];
            unit_found = true;
          }
        }
      }
      return out;
    }

    std::ostream& operator << (std::ostream& out, const Quantity& q)
    {
      out << q.m_value << '.';
      return units_power_to_stream(out,q);
    }

  } // end of namespace units
  
} // end of namespace exanb


/********************** UNIT TESTS **********************************/

#include <onika/test/unit_test.h>
#include <vector>
#include <cmath>

ONIKA_UNIT_TEST(exanb_units_conversion)
{
  using namespace onika::physics;

  // regression test data base extracted from previous version.
  struct UnitConversionCheck
  {
    double value = 0.0;
    const char* unit_str = nullptr; 
    double conv = 0.0; 
    Quantity q = {};
  };
  double x=0.0;
  std::vector<UnitConversionCheck> unit_conversion_check = {
    { x=0x1.0b0f12214fbefp-76 , "J/K" , 0x1.a9b45a3281e65p-1 , EXANB_QUANTITY( boltzmann * J / K ) } ,
    { x=0x1p+0 , "m" , 0x1.2a05f2p+33 , EXANB_QUANTITY( 1.0 * m ) } ,
    { x=0x1p+0 , "m/s" , 0x1.47ae147ae147bp-7 , EXANB_QUANTITY( 1.0 * m / s ) } ,
    { x=0x1p+0 , "ang" , 0x1p+0 , EXANB_QUANTITY( 1.0 * ang ) } ,
    { x=0x1p+0 , "J" , 0x1.98137dc066bcfp+75 , EXANB_QUANTITY( 1.0 * J ) } ,
    { x=0x1.0b0f12214fbefp-76 , "m^2*kg/s^2/K" , 0x1.a9b45a3281e65p-1 , EXANB_QUANTITY( boltzmann * (m^2) * kg / (s^2) / K ) } ,
    { x=0x1p+0 , "1/m^2" , 0x1.79ca10c924223p-67 , EXANB_QUANTITY( 1.0 / (m^2) ) } ,
    { x=0x1p+0 , "m^-2" , 0x1.79ca10c924223p-67 , EXANB_QUANTITY( 1.0 * (m^-2) ) } ,
    { x=0x1p+0 , "s/m^2" , 0x1.5798ee2308c3ap-27 , EXANB_QUANTITY( 1.0 * s / (m^2) ) } ,
    { x=0x1p+0 , "m^2/s^2" , 0x1.a36e2eb1c432cp-14 , EXANB_QUANTITY( 1.0 * (m^2) / (s^2) ) } ,
    { x=0x1p+0 , "1/s^2" , 0x1.357c299a88ea7p-80 , EXANB_QUANTITY( 1.0 / (s^2) ) } ,
    { x=0x1.37876d323b864p-37 , "C^2.s^2/m^3/kg^1" , 0x1.337f14f782bb5p-21 , EXANB_QUANTITY( epsilonZero * (C^2) * (s^2) / (m^3) / kg ) } ,
    { x=0x1.91eb851eb851fp+1 , "m" , 0x1.d3e57e8p+34 , EXANB_QUANTITY( x * m ) } ,
    { x=0x1.91eb851eb851fp+1 , "m/s" , 0x1.013a92a305532p-5 , EXANB_QUANTITY( x * m / s ) } ,
    { x=0x1.91eb851eb851fp+1 , "s" , 0x1.6d8b4ad4p+41 , EXANB_QUANTITY( x * s ) } ,
    { x=0x1.91eb851eb851fp+1 , "degree" , 0x1.c0f2ee53f24ecp-5 , EXANB_QUANTITY( x * degree ) } ,
    { x=0x1.91eb851eb851fp+1 , "ang" , 0x1.91eb851eb851fp+1 , EXANB_QUANTITY( x * ang ) } ,
    { x=0x1.91eb851eb851fp+1 , "J" , 0x1.4056fb08f47d5p+77 , EXANB_QUANTITY( x * J ) } ,
    { x=0x1.91eb851eb851fp+1 , "1/m^2" , 0x1.28908a9de5533p-65 , EXANB_QUANTITY( x / (m^2) ) } ,
    { x=0x1.91eb851eb851fp+1 , "m^-2" , 0x1.28908a9de5533p-65 , EXANB_QUANTITY( x * (m^-2) ) } ,
    { x=0x1.91eb851eb851fp+1 , "m^2/s^2" , 0x1.4940bbb1f255fp-12 , EXANB_QUANTITY( x * (m^2) / (s^2) ) } ,
    { x=0x1.91eb851eb851fp+1 , "eV" , 0x1.d9628792cb413p+14 , EXANB_QUANTITY( x * eV ) } ,
    { x=0x1.91eb851eb851fp+1 , "1/ang" , 0x1.91eb851eb851fp+1 , EXANB_QUANTITY( x / ang ) } ,
    { x=0x1.91eb851eb851fp+1 , "1/ang^2" , 0x1.91eb851eb851fp+1 , EXANB_QUANTITY( x / (ang^2) ) } ,
    { x=0x1.91eb851eb851fp+1 , "1/ang^3" , 0x1.91eb851eb851fp+1 , EXANB_QUANTITY( x / (ang^3) ) } ,
    { x=0x1.91eb851eb851fp+1 , "1/ang^4" , 0x1.91eb851eb851fp+1 , EXANB_QUANTITY( x / (ang^4) ) } ,
    { x=0x1.91eb851eb851fp+1 , "eV.ang/e-^2" , 0x1.d9628792cb413p+14 , EXANB_QUANTITY( x * eV * ang / (ec^2) ) } ,
    { x=0x1.91eb851eb851fp+1 , "C^2.s^2/m^3/kg^1" , 0x1.8cb7a343494f9p+17 , EXANB_QUANTITY( x * (C^2) * (s^2) / (m^3) / kg ) } ,
    { x=0x1.91eb851eb851fp+1 , "eV.ang/e-^2" , 0x1.d9628792cb413p+14 , EXANB_QUANTITY( x * eV * ang / (ec^2) ) } ,
    { x=-0x1.91eb851eb851fp+1 , "m" , -0x1.d3e57e8p+34 , EXANB_QUANTITY( x * m ) } ,
    { x=-0x1.91eb851eb851fp+1 , "m/s" , -0x1.013a92a305532p-5 , EXANB_QUANTITY( x * m / s ) } ,
    { x=-0x1.91eb851eb851fp+1 , "s" , -0x1.6d8b4ad4p+41 , EXANB_QUANTITY( x * s ) } ,
    { x=-0x1.91eb851eb851fp+1 , "degree" , -0x1.c0f2ee53f24ecp-5 , EXANB_QUANTITY( x * degree ) } ,
    { x=-0x1.91eb851eb851fp+1 , "ang" , -0x1.91eb851eb851fp+1 , EXANB_QUANTITY( x * ang ) } ,
    { x=-0x1.91eb851eb851fp+1 , "J" , -0x1.4056fb08f47d5p+77 , EXANB_QUANTITY( x * J ) } ,
    { x=-0x1.91eb851eb851fp+1 , "1/m^2" , -0x1.28908a9de5533p-65 , EXANB_QUANTITY( x / (m^2) ) } ,
    { x=-0x1.91eb851eb851fp+1 , "m^-2" , -0x1.28908a9de5533p-65 , EXANB_QUANTITY( x * (m^-2) ) } ,
    { x=-0x1.91eb851eb851fp+1 , "m^2/s^2" , -0x1.4940bbb1f255fp-12 , EXANB_QUANTITY( x * (m^2) / (s^2) ) } ,
    { x=-0x1.91eb851eb851fp+1 , "eV" , -0x1.d9628792cb413p+14 , EXANB_QUANTITY( x * eV ) } ,
    { x=-0x1.91eb851eb851fp+1 , "1/ang" , -0x1.91eb851eb851fp+1 , EXANB_QUANTITY( x / ang ) } ,
    { x=-0x1.91eb851eb851fp+1 , "1/ang^2" , -0x1.91eb851eb851fp+1 , EXANB_QUANTITY( x / (ang^2) ) } ,
    { x=-0x1.91eb851eb851fp+1 , "1/ang^3" , -0x1.91eb851eb851fp+1 , EXANB_QUANTITY( x / (ang^3) ) } ,
    { x=-0x1.91eb851eb851fp+1 , "1/ang^4" , -0x1.91eb851eb851fp+1 , EXANB_QUANTITY( x / (ang^4) ) } ,
    { x=-0x1.91eb851eb851fp+1 , "eV.ang/e-^2" , -0x1.d9628792cb413p+14 , EXANB_QUANTITY( x * eV * ang / (ec^2) ) } ,
    { x=-0x1.91eb851eb851fp+1 , "C^2.s^2/m^3/kg^1" , -0x1.8cb7a343494f9p+17 , EXANB_QUANTITY( x * (C^2) * (s^2) / (m^3) / kg ) } ,
    { x=-0x1.91eb851eb851fp+1 , "eV.ang/e-^2" , -0x1.d9628792cb413p+14 , EXANB_QUANTITY( x * eV * ang / (ec^2) ) } ,
    { x=0x1.999999999999ap-4 , "m" , 0x1.dcd65p+29 , EXANB_QUANTITY( x * m ) } ,
    { x=0x1.999999999999ap-4 , "m/s" , 0x1.0624dd2f1a9fcp-10 , EXANB_QUANTITY( x * m / s ) } ,
    { x=0x1.999999999999ap-4 , "s" , 0x1.74876e8p+36 , EXANB_QUANTITY( x * s ) } ,
    { x=0x1.999999999999ap-4 , "degree" , 0x1.c987103b761f5p-10 , EXANB_QUANTITY( x * degree ) } ,
    { x=0x1.999999999999ap-4 , "ang" , 0x1.999999999999ap-4 , EXANB_QUANTITY( x * ang ) } ,
    { x=0x1.999999999999ap-4 , "J" , 0x1.4675fe338564p+72 , EXANB_QUANTITY( x * J ) } ,
    { x=0x1.999999999999ap-4 , "1/m^2" , 0x1.2e3b40a0e9b4fp-70 , EXANB_QUANTITY( x / (m^2) ) } ,
    { x=0x1.999999999999ap-4 , "m^-2" , 0x1.2e3b40a0e9b4fp-70 , EXANB_QUANTITY( x * (m^-2) ) } ,
    { x=0x1.999999999999ap-4 , "m^2/s^2" , 0x1.4f8b588e368fp-17 , EXANB_QUANTITY( x * (m^2) / (s^2) ) } ,
    { x=0x1.999999999999ap-4 , "eV" , 0x1.e26e321cefc02p+9 , EXANB_QUANTITY( x * eV ) } ,
    { x=0x1.999999999999ap-4 , "1/ang" , 0x1.999999999999ap-4 , EXANB_QUANTITY( x / ang ) } ,
    { x=0x1.999999999999ap-4 , "1/ang^2" , 0x1.999999999999ap-4 , EXANB_QUANTITY( x / (ang^2) ) } ,
    { x=0x1.999999999999ap-4 , "1/ang^3" , 0x1.999999999999ap-4 , EXANB_QUANTITY( x / (ang^3) ) } ,
    { x=0x1.999999999999ap-4 , "1/ang^4" , 0x1.999999999999ap-4 , EXANB_QUANTITY( x / (ang^4) ) } ,
    { x=0x1.999999999999ap-4 , "eV.ang/e-^2" , 0x1.e26e321cefc02p+9 , EXANB_QUANTITY( x * eV * ang / (ec^2) ) } ,
    { x=0x1.999999999999ap-4 , "C^2.s^2/m^3/kg^1" , 0x1.944c448c513bdp+12 , EXANB_QUANTITY( x * (C^2) * (s^2) / (m^3) / kg ) } ,
    { x=0x1.999999999999ap-4 , "eV.ang/e-^2" , 0x1.e26e321cefc02p+9 , EXANB_QUANTITY( x * eV * ang / (ec^2) ) } ,
    { x=-0x1.999999999999ap-4 , "m" , -0x1.dcd65p+29 , EXANB_QUANTITY( x * m ) } ,
    { x=-0x1.999999999999ap-4 , "m/s" , -0x1.0624dd2f1a9fcp-10 , EXANB_QUANTITY( x * m / s ) } ,
    { x=-0x1.999999999999ap-4 , "s" , -0x1.74876e8p+36 , EXANB_QUANTITY( x * s ) } ,
    { x=-0x1.999999999999ap-4 , "degree" , -0x1.c987103b761f5p-10 , EXANB_QUANTITY( x * degree ) } ,
    { x=-0x1.999999999999ap-4 , "ang" , -0x1.999999999999ap-4 , EXANB_QUANTITY( x * ang ) } ,
    { x=-0x1.999999999999ap-4 , "J" , -0x1.4675fe338564p+72 , EXANB_QUANTITY( x * J ) } ,
    { x=-0x1.999999999999ap-4 , "1/m^2" , -0x1.2e3b40a0e9b4fp-70 , EXANB_QUANTITY( x / (m^2) ) } ,
    { x=-0x1.999999999999ap-4 , "m^-2" , -0x1.2e3b40a0e9b4fp-70 , EXANB_QUANTITY( x * (m^-2) ) } ,
    { x=-0x1.999999999999ap-4 , "m^2/s^2" , -0x1.4f8b588e368fp-17 , EXANB_QUANTITY( x * (m^2) / (s^2) ) } ,
    { x=-0x1.999999999999ap-4 , "eV" , -0x1.e26e321cefc02p+9 , EXANB_QUANTITY( x * eV ) } ,
    { x=-0x1.999999999999ap-4 , "1/ang" , -0x1.999999999999ap-4 , EXANB_QUANTITY( x / ang ) } ,
    { x=-0x1.999999999999ap-4 , "1/ang^2" , -0x1.999999999999ap-4 , EXANB_QUANTITY( x / (ang^2) ) } ,
    { x=-0x1.999999999999ap-4 , "1/ang^3" , -0x1.999999999999ap-4 , EXANB_QUANTITY( x / (ang^3) ) } ,
    { x=-0x1.999999999999ap-4 , "1/ang^4" , -0x1.999999999999ap-4 , EXANB_QUANTITY( x / (ang^4) ) } ,
    { x=-0x1.999999999999ap-4 , "eV.ang/e-^2" , -0x1.e26e321cefc02p+9 , EXANB_QUANTITY( x * eV * ang / (ec^2) ) } ,
    { x=-0x1.999999999999ap-4 , "C^2.s^2/m^3/kg^1" , -0x1.944c448c513bdp+12 , EXANB_QUANTITY( x * (C^2) * (s^2) / (m^3) / kg ) } ,
    { x=-0x1.999999999999ap-4 , "eV.ang/e-^2" , -0x1.e26e321cefc02p+9 , EXANB_QUANTITY( x * eV * ang / (ec^2) ) } ,
  };
  for(const auto & test : unit_conversion_check)
  {
    const Quantity qa = make_quantity( test.value , test.unit_str );
    const double a = qa.convert();
    const double b = test.q.convert();
    ONIKA_TEST_ASSERT( a==b || ( std::fabs(a-b) / std::max( std::fabs(a) , std::fabs(b) ) ) < 1.e-6 );
  }


# define TEST_EXPR( v , u , expr ) \
{ \
  const Quantity qa = make_quantity(v,u); \
  const double a = qa.convert(); \
  const Quantity qb = EXANB_QUANTITY( expr ); \
  const double b = qb.convert(); \
  ONIKA_TEST_ASSERT( a==b || ( std::fabs(a-b) / std::max( std::fabs(a) , std::fabs(b) ) ) < 1.e-6 ); \
}

  TEST_EXPR( boltzmann, "J/K" , boltzmann * J / K );
  TEST_EXPR( 1.0 , "m" , 1.0 * m );
  TEST_EXPR( 1.0 , "m/s" , 1.0 * m / s );
  TEST_EXPR( 1.0 , "ang" , 1.0 * ang );
  TEST_EXPR( 1.0 , "J" , 1.0 * J );
  TEST_EXPR( boltzmann , "m^2*kg/s^2/K" , boltzmann * (m^2) * kg / (s^2) / K );
  TEST_EXPR( 1.0, "1/m^2" , 1.0 / (m^2) );
  TEST_EXPR( 1.0, "m^-2" , 1.0 * (m^-2) );
  TEST_EXPR( 1.0, "s/m^2" , 1.0 * s / (m^2) );
  TEST_EXPR( 1.0 , "m^2/s^2" , 1.0 * (m^2) / (s^2) );
  TEST_EXPR( 1.0 , "1/s^2" , 1.0 / (s^2) );
  TEST_EXPR( epsilonZero , "C^2.s^2/m^3/kg^1" , epsilonZero * (C^2) * (s^2) / (m^3) / kg );
  
  for( double x : std::vector<double>{ 3.14 , -3.14 , 0.1 , -0.1 } )
  {
    TEST_EXPR( x , "m" , x * m );
    TEST_EXPR( x , "m/s" , x * m / s );
    TEST_EXPR( x , "s" , x * s );
    TEST_EXPR( x , "degree" , x * degree );
    TEST_EXPR( x , "ang" , x * ang );
    TEST_EXPR( x , "J" , x * J );
    TEST_EXPR( x , "1/m^2" , x / (m^2) );
    TEST_EXPR( x , "m^-2" , x * (m^-2) );
    TEST_EXPR( x , "m^2/s^2" , x * (m^2) / (s^2) );
    TEST_EXPR( x , "eV" , x * eV );
    TEST_EXPR( x , "1/ang" , x / ang );
    TEST_EXPR( x , "1/ang^2" , x / (ang^2) );
    TEST_EXPR( x , "1/ang^3" , x / (ang^3) );
    TEST_EXPR( x , "1/ang^4" , x / (ang^4) );
    TEST_EXPR( x , "eV.ang/e-^2" , x * eV * ang / (ec^2) );
    TEST_EXPR( x , "C^2.s^2/m^3/kg^1" , x * (C^2) * (s^2) / (m^3) / kg );
    TEST_EXPR( x, "eV.ang/e-^2" , x * eV * ang / (ec^2) );
  }

}

