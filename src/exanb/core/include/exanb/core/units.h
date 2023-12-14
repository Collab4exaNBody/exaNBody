#pragma once

#include <limits>
#include <cmath>
#include <string>

#include <exanb/core/physics_constants.h>

// #define EXANB_LEGACY_UNITS_DEPRECATED 1 // will be set soon ...

namespace exanb
{

  namespace units
  {

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
      NUMBER_OF_UNIT_CLASSES,
      OTHER = -1
    };
      
    static inline constexpr auto elementaryChargeCoulomb = legacy_constant::elementaryCharge;
    static inline constexpr auto undefined_value = std::numeric_limits<double>::quiet_NaN();
    
    struct UnitDefinition
    {
      UnitClass m_class = OTHER;
      double m_to_si = 1.0;
      const char* m_short_name = "?";
      const char* m_name = "unknow";
      
      inline bool operator == (const UnitDefinition& other) const { return m_class==other.m_class && m_to_si==other.m_to_si; }      
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

    namespace symbols
    {
      static inline constexpr auto m = meter;
      static inline constexpr auto mm = millimeter;
      static inline constexpr auto um = micron;
      static inline constexpr auto nm = nanometer;
      static inline constexpr auto ang = angstrom;
      
      static inline constexpr auto kg = kilogram;
      static inline constexpr auto g = gram;
      static inline constexpr auto Da = atomic_mass_unit;
      
      static inline constexpr auto h = hour;
      static inline constexpr auto s = second;
      static inline constexpr auto us = microsecond;
      static inline constexpr auto ns = nanosecond;
      static inline constexpr auto ps = picosecond;
      static inline constexpr auto fs = fetosecond;
      
      static inline constexpr auto C = coulomb;
      static inline constexpr auto ec = elementary_charge;
      
      static inline constexpr auto K = kelvin;
      
      static inline constexpr auto mol = ::exanb::units::mol;

      static inline constexpr auto cd = candela;
      
      static inline constexpr auto rad = radian;
      static inline constexpr auto degree = ::exanb::units::degree;
      
      static inline constexpr auto J = joule;
      static inline constexpr auto eV = electron_volt;
      static inline constexpr auto cal = calorie;
      static inline constexpr auto kcal = kcalorie;
    }

    struct UnitSystem
    {
      UnitDefinition m_units[NUMBER_OF_UNIT_CLASSES];
    };

    struct UnitPowers
    {
      double m_powers[NUMBER_OF_UNIT_CLASSES] = { 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. };
    };

/*
    namespace symbols
    {
      using m = meter;
    }
*/
      
    // list of all definitions
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
      
    static inline constexpr int number_of_units = sizeof(all_units) / sizeof(UnitDefinition);
    
    inline UnitDefinition unit_from_symbol( const std::string& s )
    {
      if( s.empty() ) return no_unity;
      for(int i=0;i<number_of_units;i++) if( s == all_units[i].m_short_name ) return all_units[i];
      return unknown;
    }

    static inline constexpr UnitSystem SI =
    { { meter
      , kilogram
      , second
      , coulomb
      , kelvin
      , mol
      , candela
      , radian
      , joule } };

  }

}

#include "exanb/internal_units.h"

namespace exanb
{

  namespace units
  {
    static inline constexpr UnitSystem internal_unit_system =
    { { INTERNAL_UNIT_LENGTH,
        INTERNAL_UNIT_MASS,
        INTERNAL_UNIT_TIME,
        INTERNAL_UNIT_ELECTRIC_CURRENT,
        INTERNAL_UNIT_TEMPERATURE,
        INTERNAL_UNIT_AMOUNT_OF_SUBSTANCE,
        INTERNAL_UNIT_LUMINOUS_INTENSITY,
        INTERNAL_UNIT_ANGLE,
        joule } };
      
    struct Quantity
    {
      double m_value = 0.0;
      UnitSystem m_system = SI;
      UnitPowers m_unit_powers = { { 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. } };

      inline constexpr double convert( const UnitSystem& other = internal_unit_system ) const
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
      
      inline operator double() const { return convert(); }
    };

    inline Quantity operator * ( const Quantity& qlhs , const Quantity& qrhs )
    {
      Quantity q = qrhs;
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
            }
            q.m_value *= pow( qrhs.m_system.m_units[i].m_to_si / q.m_system.m_units[i].m_to_si , qrhs.m_unit_powers.m_powers[i] );
            q.m_unit_powers.m_powers[i] += qrhs.m_unit_powers.m_powers[i];
          }
        }
        else
        {
          q.m_system.m_units[i] = unknown; // will popup a nan
          q.m_unit_powers.m_powers[i] = 1.0;
        }
      }
      return q;
    }

    inline Quantity operator ^ ( const Quantity& qlhs , double power )
    {
      Quantity q = qlhs;
      q.m_value = pow( q.m_value , power );
      for(int i=0;i<NUMBER_OF_UNIT_CLASSES;i++)
      {
        q.m_unit_powers.m_powers[i] *= power;
      }
      return q;
    }

    inline Quantity dispatch_energy_units( const Quantity& q )
    {
      double joule_factor = pow( q.m_system.m_units[ENERGY].m_to_si , q.m_unit_powers.m_powers[ENERGY] );
      Quantity qnrj = { joule_factor , SI };
      qnrj.m_unit_powers.m_powers[LENGTH] = q.m_unit_powers.m_powers[ENERGY] * 2;
      qnrj.m_unit_powers.m_powers[MASS  ] = q.m_unit_powers.m_powers[ENERGY] * 1;
      qnrj.m_unit_powers.m_powers[TIME]   = q.m_unit_powers.m_powers[ENERGY] * -2;
      Quantity qremain = q;
      qremain.m_unit_powers.m_powers[ENERGY] = 0.0;
      return qremain * qnrj;
    }

    inline Quantity operator * ( double value , const UnitDefinition& U )
    {
      Quantity q = { value };
      q.m_system.m_units[ U.m_class ] = U;
      q.m_unit_powers.m_powers[ U.m_class ] = 1.0;
      return dispatch_energy_units(q);
    }

    inline Quantity operator ^ ( const UnitDefinition& U , double power )
    {
      return dispatch_energy_units( 1.0 * U ) ^ power;
    }

    inline Quantity operator * ( double value , const Quantity& qrhs )
    {
      Quantity q = qrhs;
      q.m_value *= value;
      return dispatch_energy_units(q);
    }

    inline Quantity operator * ( const Quantity& qlhs , const UnitDefinition& U )
    {
      return qlhs * dispatch_energy_units( 1.0 * U );
    }

    inline Quantity operator / ( const Quantity& qlhs , const Quantity& qrhs )
    {
      return qlhs * ( qrhs ^ -1.0 );
    }

    template<class StreamT>
    inline StreamT& units_power_to_stream (StreamT& out, const Quantity& q)
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

    // pretty printing
    template<class StreamT>
    inline StreamT& operator << (StreamT& out, const Quantity& q)
    {
      out << q.m_value << ' ';
      return units_power_to_stream(out,q);
    }

    Quantity make_quantity( double value , const std::string& units_and_powers );
    Quantity quantity_from_string(const std::string& s, bool& conversion_done);
    Quantity quantity_from_string(const std::string& s);

  }

  /************ backward compatibility with older versions ***********/  
  struct UnityConverterHelper
  {
#   ifdef EXANB_LEGACY_UNITS_DEPRECATED
    [[deprecated]]
#   endif
    static inline double convert( double value, const std::string& units_and_powers )
    {
      return units::make_quantity(value,units_and_powers).convert();
    }
  };
  using units::Quantity;
  using units::make_quantity;
   
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


