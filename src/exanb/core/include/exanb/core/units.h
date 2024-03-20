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

#include <limits>
#include <cmath>
#include <string>

#include <exanb/core/physics_constants.h>
#include <onika/cuda/cuda.h>

#define EXANB_UNITS_V2 1
//#define EXANB_UNITS_DEPRECATED [[deprecated]]

#ifdef EXANB_UNITS_DEPRECATED
#define EXANB_LEGACY_UNITS_DEPRECATED 1
#else
#define EXANB_UNITS_DEPRECATED /**/
#endif

namespace exanb
{

  namespace units
  {

    /*
     * Defines different kinds of units.
     * ENERGY is a special kind, not stored in quantities, as it is converted to length^2 * mass / time^2
     */
    enum UnitClass
    {
      LENGTH,
      MASS,
      TIME,
      CHARGE,
      TEMP,
      AMOUNT,
      LUMINOUS,
      ANGLE,
      ENERGY,
      NUMBER_OF_UNIT_CLASSES = ENERGY,
      OTHER = -1
    };
    
    /*
     * unit classes as strings
     */
    static inline constexpr const char* g_unit_class_str[NUMBER_OF_UNIT_CLASSES+1] = {
      "LENGTH",
      "MASS",
      "TIME",
      "CHARGE",
      "TEMP",
      "AMOUNT",
      "LUMINOUS",
      "ANGLE",
      "ENERGY"
    };
    
    // shortcuts to particular values
    static inline constexpr auto elementaryChargeCoulomb = legacy_constant::elementaryCharge;
    static inline constexpr auto undefined_value = std::numeric_limits<double>::quiet_NaN();
    
    /*
     * A unit defined by its class (length, mass, etc.) ,
     * its conversion factor to its IS system counterpart
     * and its full and short name.
     */
    struct UnitDefinition
    {
      UnitClass m_class = OTHER;
      double m_to_si = 1.0;
      const char* m_short_name = "?";
      const char* m_name = "unknow";
      ONIKA_HOST_DEVICE_FUNC inline bool operator == (const UnitDefinition& other) const { return m_class==other.m_class && m_to_si==other.m_to_si; }
      inline const char* short_name() const { return m_short_name; }
      inline const char* name() const { return m_name; }
    };

    /*********************************************************************/
    /***************** All units available *******************************/
    /*********************************************************************/

    //length
    static inline constexpr UnitDefinition meter              = { LENGTH   , 1.0                     , "m"        , "meter" };
    static inline constexpr UnitDefinition millimeter         = { LENGTH   , 1.0e-3                  , "mm"       , "millimeter" };
    static inline constexpr UnitDefinition micron             = { LENGTH   , 1.0e-6                  , "um"       , "micron" };
    static inline constexpr UnitDefinition nanometer          = { LENGTH   , 1.0e-9                  , "nm"       , "nanometer" };
    static inline constexpr UnitDefinition angstrom           = { LENGTH   , 1.0e-10                 , "ang"      , "angstrom" };
    
    //mass
    static inline constexpr UnitDefinition kilogram           = { MASS     , 1.0                     , "kg"       , "kilogram" };
    static inline constexpr UnitDefinition gram               = { MASS     , 1.0e-3                  , "g"        , "gram" };
    static inline constexpr UnitDefinition atomic_mass_unit   = { MASS     , 1.660539040e-27         , "Da"       , "Dalton" };
    
    //time
    static inline constexpr UnitDefinition hour               = { TIME     , 3600                    , "h"        , "hour" };
    static inline constexpr UnitDefinition second             = { TIME     , 1.0                     , "s"        , "second" };
    static inline constexpr UnitDefinition microsecond        = { TIME     , 1.0e-6                  , "us"       , "microsecond" };
    static inline constexpr UnitDefinition nanosecond         = { TIME     , 1.0e-9                  , "ns"       , "nanosecond" };
    static inline constexpr UnitDefinition picosecond         = { TIME     , 1.0e-12                 , "ps"       , "picosecond" };
    static inline constexpr UnitDefinition fetosecond         = { TIME     , 1.0e-15                 , "fs"       , "fetosecond" };

    //Electric current
    static inline constexpr UnitDefinition coulomb            = { CHARGE   , 1.0                     , "C"        , "coulomb" };
    static inline constexpr UnitDefinition elementary_charge  = { CHARGE   , elementaryChargeCoulomb , "e-"       , "elementary_charge" };

    //temperature
    static inline constexpr UnitDefinition kelvin             = { TEMP     , 1.0                     , "K"        , "kelvin" };
    static inline constexpr UnitDefinition celsius            = { TEMP     , 1.0                     , "°"        , "celsius" }; // should never be used

    //amount of substance
    static inline constexpr UnitDefinition mol                = { AMOUNT   , 1.0                     , "mol"      , "mol" };
    static inline constexpr UnitDefinition particle           = { AMOUNT   , (1/6.02214076)*1.0e23   , "particle" , "particle" };

    //luminous intensity
    static inline constexpr UnitDefinition candela            = { LUMINOUS , 1.0                     , "cd"       , "candela" };

    // angle
    static inline constexpr UnitDefinition radian             = { ANGLE    , 1.0                     , "rad"      , "radian" };
    static inline constexpr UnitDefinition degree             = { ANGLE    , M_PI/180.0              , "degree"   , "degree" };

    // energy
    static inline constexpr UnitDefinition joule              = { ENERGY   , 1.0                     , "J"        , "joule" };
    static inline constexpr UnitDefinition electron_volt      = { ENERGY   , elementaryChargeCoulomb , "eV"       , "electron_volt" };
    static inline constexpr UnitDefinition calorie            = { ENERGY   , 4.1868                  , "cal"      , "calorie" };
    static inline constexpr UnitDefinition kcalorie           = { ENERGY   , 4186.8                  , "kcal"     , "kcalorie" };
    
    // unit less
    static inline constexpr UnitDefinition no_unity           = { OTHER    , 1.0                     , ""         , "" };

    // unknown
    static inline constexpr UnitDefinition unknown            = { OTHER    , undefined_value         , ""         , "unknow" };

    /*********************************************************************/
    /*********************************************************************/

    /*
     * A UnitSystem is a set of units, one per type (class) of unit
     */
    struct UnitSystem
    {
      UnitDefinition m_units[NUMBER_OF_UNIT_CLASSES];      
      inline const UnitDefinition& length() const { return m_units[LENGTH]; }
      inline const UnitDefinition& time() const { return m_units[TIME]; }
      inline const UnitDefinition& charge() const { return m_units[CHARGE]; }
      inline const UnitDefinition& temperature() const { return m_units[TEMP]; }
      inline const UnitDefinition& amount() const { return m_units[AMOUNT]; }
      inline const UnitDefinition& luminance() const { return m_units[LUMINOUS]; }
      inline const UnitDefinition& angle() const { return m_units[ANGLE]; }
      inline const UnitDefinition& energy() const { return m_units[ENERGY]; }
    };

    /*
     * powers associated with units of each class 
     */
    struct UnitPowers
    {
      double m_powers[NUMBER_OF_UNIT_CLASSES] = { 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. };
    };
      
    // list of all available unit definitions
    static inline constexpr UnitDefinition all_units[] = {
      meter, millimeter, micron, nanometer, angstrom,
      kilogram, gram, atomic_mass_unit,
      hour, second, microsecond, nanosecond, picosecond, fetosecond,
      coulomb, elementary_charge,
      kelvin, celsius,
      mol, particle,
      candela,
      radian, degree,
      joule, electron_volt, calorie, kcalorie };
    
    // number of available unit definitions 
    static inline constexpr int number_of_units = sizeof(all_units) / sizeof(UnitDefinition);
    
    /*
     * return a unit definition given a unit's short name
     */
    inline UnitDefinition unit_from_symbol( const std::string& s )
    {
      if( s.empty() ) return no_unity;
      for(int i=0;i<number_of_units;i++) if( s == all_units[i].m_short_name ) return all_units[i];
      return unknown;
    }

    /*
     * International System units
     */
    static inline constexpr UnitSystem SI =
    { { meter
      , kilogram
      , second
      , coulomb
      , kelvin
      , mol
      , candela
      , radian } };

  }

  /************ backward compatibility layer ***********/  
  struct EXANB_UNITS_DEPRECATED EnumUnities
  {
    units::UnitDefinition m_unit;
    inline constexpr EnumUnities(const units::UnitDefinition& u) : m_unit(u) {}
    inline constexpr operator units::UnitDefinition() const { return m_unit; }
    static inline constexpr units::UnitDefinition meter = units::meter;
    static inline constexpr units::UnitDefinition angstrom = units::angstrom;
    static inline constexpr units::UnitDefinition atomic_mass_unit = units::atomic_mass_unit;
    static inline constexpr units::UnitDefinition gram = units::gram;
    static inline constexpr units::UnitDefinition picosecond = units::picosecond;
    static inline constexpr units::UnitDefinition second = units::second;
    static inline constexpr units::UnitDefinition elementary_charge = units::elementary_charge;
    static inline constexpr units::UnitDefinition kelvin = units::kelvin;
    static inline constexpr units::UnitDefinition particle = units::particle;
    static inline constexpr units::UnitDefinition candela = units::candela;
    static inline constexpr units::UnitDefinition radian = units::radian;
  };
  /************ END OF backward compatibility *********************/

}

#include "exanb/internal_units.h"

namespace exanb
{

  namespace units
  {
  
    /*
     * Internal unit system, as defined by application through its exanb/internal_units.h header file
     */
    static inline constexpr UnitSystem internal_unit_system =
    { { INTERNAL_UNIT_LENGTH,
        INTERNAL_UNIT_MASS,
        INTERNAL_UNIT_TIME,
        INTERNAL_UNIT_ELECTRIC_CURRENT,
        INTERNAL_UNIT_TEMPERATURE,
        INTERNAL_UNIT_AMOUNT_OF_SUBSTANCE,
        INTERNAL_UNIT_LUMINOUS_INTENSITY,
        INTERNAL_UNIT_ANGLE} };
      
    /*
     * A Quantity is a value expressed with a certain set of units (at most one unit per unit class) and associated powers
     * Quantity has an implicit conversion operator to double, wich converts expressed value to application's internal unit system
     * Exemple :
     * using namespace exanb::units;
     * Quantity q = EXANB_QUANTITY( 3.2 * (g^3) / s ); // EXANB_QUANTITY macro allows to use short name unit definitons in online expressions
     * Quantity r = 3.2 * (gram^3) / second; // same as above
     * double a = q;
     * double b = q.convert(); // same as above
     * double c = q.convert( internal_unit_system ); // same as above
     * double x = q.convert( SI ); // converts to Internal System units based value
     */
    struct Quantity
    {
      double m_value = 0.0;
      UnitSystem m_system = SI;
      UnitPowers m_unit_powers = { { 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. } };

      ONIKA_HOST_DEVICE_FUNC inline double convert( const UnitSystem& other ) const
      {
        double value = m_value;
        for(int i=0;i<NUMBER_OF_UNIT_CLASSES;i++)
        {
          if( m_system.m_units[i].m_class != i ) // invalid units or internal error
          {
            return undefined_value;
          } 
          if( m_system.m_units[i].m_to_si != other.m_units[i].m_to_si )
          {
            value *= pow( m_system.m_units[i].m_to_si / other.m_units[i].m_to_si , m_unit_powers.m_powers[i] );
          }
        }
        return value;      
      }
      ONIKA_HOST_DEVICE_FUNC inline double convert() const
      {
        constexpr UnitSystem IUS = ::exanb::units::internal_unit_system;
        return convert( IUS );
      }
      
      ONIKA_HOST_DEVICE_FUNC inline operator double() const { return convert(); }
    };

    /*
     * Multiplies 2 quantities.
     * Converts qrhs units to the units used in qlhs if needed.
     */
    ONIKA_HOST_DEVICE_FUNC inline Quantity operator * ( const Quantity& qlhs , const Quantity& qrhs )
    {
      Quantity q = qlhs;
      q.m_value *= qrhs.m_value;
      for(int i=0;i<NUMBER_OF_UNIT_CLASSES;i++)
      {
        if( ( q.m_system.m_units[i].m_class == i ) && ( q.m_system.m_units[i].m_class == qrhs.m_system.m_units[i].m_class ) )
        {
          if( qrhs.m_unit_powers.m_powers[i] != 0.0 )
          {
            if( q.m_unit_powers.m_powers[i] == 0.0 )
            {
              q.m_system.m_units[i] = qrhs.m_system.m_units[i];
              q.m_unit_powers.m_powers[i] = qrhs.m_unit_powers.m_powers[i];
            }
            else
            {
              q.m_value *= pow( qrhs.m_system.m_units[i].m_to_si / q.m_system.m_units[i].m_to_si , qrhs.m_unit_powers.m_powers[i] );
              q.m_unit_powers.m_powers[i] += qrhs.m_unit_powers.m_powers[i];
            }
          }
        }
        else
        {
          q.m_system.m_units[i] = unknown; // will popup a nan
          q.m_unit_powers.m_powers[i] = 0.0;
        }
      }
      return q;
    }

    /*
     * Raise a quantity to a power.
     * For instance, if q = 2 * m * (s^-2) then q^2 = 4 * (m^2) * (s^-4)
     */
    ONIKA_HOST_DEVICE_FUNC inline Quantity operator ^ ( const Quantity& qlhs , double power )
    {
      Quantity q = qlhs;
      q.m_value = pow( q.m_value , power );
      for(int i=0;i<NUMBER_OF_UNIT_CLASSES;i++)
      {
        q.m_unit_powers.m_powers[i] *= power;
      }
      return q;
    }

    /*
     * When a scalar value mutliplies a UnitDefinition, it builds up a quantity
     */
    ONIKA_HOST_DEVICE_FUNC inline Quantity operator * ( double value , const UnitDefinition& U )
    {
      Quantity q = { value };
      if( U.m_class>=0 && U.m_class < NUMBER_OF_UNIT_CLASSES )
      {
        q.m_system.m_units[ U.m_class ] = U;
        q.m_unit_powers.m_powers[ U.m_class ] = 1.0;
      }
      // energy is dispatched to other units
      else if( U.m_class == ENERGY )
      {
        q.m_value *= U.m_to_si;
        q.m_system.m_units[ LENGTH ] = meter;
        q.m_unit_powers.m_powers[ LENGTH ] = 2;
        q.m_system.m_units[ MASS ] = kilogram;
        q.m_unit_powers.m_powers[ MASS ] = 1;
        q.m_system.m_units[ TIME ] = second;
        q.m_unit_powers.m_powers[ TIME ] = -2;        
      }
      else
      {
        q.m_value = undefined_value;
      }
      return q;
    }

    /*
     * When a UnitDefinition is raised to a power, it builds up a quantity with value=1.0
     */
    ONIKA_HOST_DEVICE_FUNC inline Quantity operator ^ ( const UnitDefinition& U , double power )
    {
      return ( 1.0 * U ) ^ power;
    }

    /*
     * Scalar multiplication of a quantity.
     * Exemple : 5.0 * ( 1.0 * m * (s^-2) ) = 5.0 * m * (s^-2)
     */
    ONIKA_HOST_DEVICE_FUNC inline Quantity operator * ( double value , const Quantity& qrhs )
    {
      Quantity q = qrhs;
      q.m_value *= value;
      return q;
    }

    /*
     * equivalent to building a quantity from 1.0 * UnitDefinition, and the nmultiplying the two quantities
     */
    ONIKA_HOST_DEVICE_FUNC inline Quantity operator * ( const Quantity& qlhs , const UnitDefinition& U )
    {
      return qlhs * ( 1.0 * U );
    }

    /*
     * qa / qb = qa * ( qb^-1 )
     */
    ONIKA_HOST_DEVICE_FUNC inline Quantity operator / ( const Quantity& qlhs , const Quantity& qrhs )
    {
      return qlhs * ( qrhs ^ -1.0 );
    }

    /*
     * qa / U = qa * ( ( 1.0 * U ) ^-1 )
     */
    ONIKA_HOST_DEVICE_FUNC inline Quantity operator / ( const Quantity& qlhs , const UnitDefinition& U )
    {
      return qlhs * ( ( 1.0 * U ) ^ -1.0 );
    }

    /*
     * x / U = x * ( ( 1.0 * U ) ^-1 )
     */
    ONIKA_HOST_DEVICE_FUNC inline Quantity operator / ( double x , const UnitDefinition& U )
    {
      return x * ( ( 1.0 * U ) ^ -1.0 );
    }

    /*
     * x / q = x * ( q^-1 )
     */
    ONIKA_HOST_DEVICE_FUNC inline Quantity operator / ( double x , const Quantity& qrhs )
    {
      return x * ( qrhs ^ -1.0 );
    }

    // pretty printing
    std::ostream& units_power_to_stream (std::ostream& out, const Quantity& q);
    std::ostream& operator << (std::ostream& out, const exanb::units::Quantity& q);

    // utulity functions to builds quantities from strings (i.e. YAML)
    Quantity make_quantity( double value , const std::string& units_and_powers );
    Quantity quantity_from_string(const std::string& s, bool& conversion_done);
    Quantity quantity_from_string(const std::string& s);
  }

  /************ backward compatibility layer ***********/  
  struct EXANB_UNITS_DEPRECATED UnityConverterHelper
  {
    static inline double convert( double value, const std::string& units_and_powers )
    {
      return units::make_quantity(value,units_and_powers).convert();
    }
  };
  using units::Quantity;
  using units::make_quantity;
  /************ END OF backward compatibility *********************/  

}

/***************** YAML conversion *****************/

#include <yaml-cpp/yaml.h>
#include <sstream>

namespace YAML
{
  template<> struct convert< exanb::units::Quantity >
  {
    static inline Node encode(const exanb::units::Quantity& q)
    {
      std::ostringstream oss;
      oss << q;
      Node node;
      node = oss.str();
      return node;
    }

    static inline bool decode(const Node& node, exanb::units::Quantity& q)
    {
      if( node.IsScalar() ) // value and unit in the same string
      {
        bool conv_ok = false;
        q = exanb::units::quantity_from_string( node.as<std::string>() , conv_ok );
        return conv_ok;
      }
      else if( node.IsMap() )
      {
        q = exanb::units::make_quantity( node["value"].as<double>() , node["unity"].as<std::string>() );
        return true;
      }
      else
      {
        return false;
      }
    }

  };

}

// allow use of quantities without parsing strings
#define EXANB_QUANTITY( __expr ) [&]() -> ::exanb::units::Quantity { \
  using namespace ::exanb::units; \
  [[maybe_unused]] constexpr auto m = meter; \
  [[maybe_unused]] constexpr auto mm = millimeter; \
  [[maybe_unused]] constexpr auto um = micron; \
  [[maybe_unused]] constexpr auto nm = nanometer; \
  [[maybe_unused]] constexpr auto ang = angstrom; \
  [[maybe_unused]] constexpr auto kg = kilogram; \
  [[maybe_unused]] constexpr auto g = gram; \
  [[maybe_unused]] constexpr auto Da = atomic_mass_unit; \
  [[maybe_unused]] constexpr auto h = hour; \
  [[maybe_unused]] constexpr auto s = second; \
  [[maybe_unused]] constexpr auto us = microsecond; \
  [[maybe_unused]] constexpr auto ns = nanosecond; \
  [[maybe_unused]] constexpr auto ps = picosecond; \
  [[maybe_unused]] constexpr auto fs = fetosecond; \
  [[maybe_unused]] constexpr auto C = coulomb; \
  [[maybe_unused]] constexpr auto ec = elementary_charge; \
  [[maybe_unused]] constexpr auto K = kelvin; \
  [[maybe_unused]] constexpr auto mol = ::exanb::units::mol; \
  [[maybe_unused]] constexpr auto cd = candela; \
  [[maybe_unused]] constexpr auto rad = radian; \
  [[maybe_unused]] constexpr auto degree = ::exanb::units::degree;       \
  [[maybe_unused]] constexpr auto J = joule; \
  [[maybe_unused]] constexpr auto eV = electron_volt; \
  [[maybe_unused]] constexpr auto cal = calorie; \
  [[maybe_unused]] constexpr auto kcal = kcalorie; \
  return ( __expr ); \
}()

