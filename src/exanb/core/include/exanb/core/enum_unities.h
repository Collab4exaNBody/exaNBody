#pragma once

#include <limits>
#include <cmath>

#include <exanb/core/physics_constants.h>

namespace exanb
{

  namespace unit
  {
  
    struct UnitDefinition
    {
      double m_to_si = 1.0;
      const char* m_short_name = "?";
      const char* m_name = "unknow";
    };

    struct UnitWithPower
    {
      UnitDefinition m_unit = {};
      int m_power = 1;
      int m_power_div = 1;
    };
      
    struct UnitVector
    {
      static constexpr int MAX_UNITS = 16;
      UnitWithPower m_units[MAX_UNITS];
      int m_unit_count;
      inline constexpr double to_si( double value = 1.0 )
      {
        for(int i=0;i<m_unit_count;i++) value *= pow( m_units[i].m_unit.m_to_si , m_units[i].m_power * 1.0 / m_units[i].m_power_div );
        return value;
      }
      inline constexpr double to( const UnitVector& uv , double value = 1.0 )
      {
        return to_si(value) / uv.to_si();
      }
    };

    namespace definitions
    {
      //length
      static inline constexpr UnitDefinition m        = { 1.0                               , "m"        , "meter" };
      static inline constexpr UnitDefinition mm       = { 1.0e-3                            , "mm"       , "millimeter" };
      static inline constexpr UnitDefinition um       = { 1.0e-6                            , "um"       , "micron" };
      static inline constexpr UnitDefinition nm       = { 1.0e-9                            , "nm"       , "nanometer" };
      static inline constexpr UnitDefinition ang      = { 1.0e-10                           , "ang"      , "angstrom" };
      
      //mass
      static inline constexpr UnitDefinition kg       = { 1.0                               , "kg"       , "kilogram" };
      static inline constexpr UnitDefinition g        = { 1.0e-3                            , "g"        , "gram" };
      static inline constexpr UnitDefinition Da       = { 1.660539040e-27                   , "Da"       , "Dalton" };
      
      //time
      static inline constexpr UnitDefinition h        = { 3600                              , "h"        , "hour" };
      static inline constexpr UnitDefinition s        = { 1.0                               , "s"        , "second" };
      static inline constexpr UnitDefinition us       = { 1.0e-6                            , "us"       , "microsecond" };
      static inline constexpr UnitDefinition ns       = { 1.0e-9                            , "ns"       , "nanosecond" };
      static inline constexpr UnitDefinition ps       = { 1.0e-12                           , "ps"       , "picosecond" };
      static inline constexpr UnitDefinition fs       = { 1.0e-15                           , "fs"       , "fetosecond" };

      //Electric current
      static inline constexpr UnitDefinition C        = { 1.0                               , "C"        , "coulomb" };
      static inline constexpr UnitDefinition ec       = { legacy_constant::elementaryCharge , "e-"       , "elementary_charge" };

      //temperature
      static inline constexpr UnitDefinition K        = { 1.0                               , "K"        , "kelvin" };
      static inline constexpr UnitDefinition degC     = { 1.0                               , "°"        , "celsius" }; // should never be used

      //amount of substance
      static inline constexpr UnitDefinition mol      = { 1.0                               , "mol"      , "mol" };
      static inline constexpr UnitDefinition particle = { (1/6.02214076)*1.0e23             , "particle" , "particle" };

      //luminous intensity
      static inline constexpr UnitDefinition cd       = { 1.0                               , "cd"       , "candela" };

      //others
      static inline constexpr UnitDefinition J        = { 1.0                               , "J"        , "joule" };
      static inline constexpr UnitDefinition eV       = { legacy_constant::elementaryCharge , "eV"       , "electron_volt" };
      static inline constexpr UnitDefinition cal      = { 4.1868                            , "cal"      , "calorie" };
      static inline constexpr UnitDefinition kcal     = { 4186.8                            , "kcal"     , "kcalorie" };

      // angle
      static inline constexpr UnitDefinition rad      = { 1.0                               , "rad"      , "radian" };
      static inline constexpr UnitDefinition degree   = { M_PI/180.0                        , "degree"   , "degree" };
      
      // unit less
      static inline constexpr UnitDefinition no_unity = { 1.0                               , ""         , "" };

      // unknown
      static inline constexpr UnitDefinition unknow = { std::numeric_limits<T>::quiet_NaN() , ""         , "unknow" };
      
      // list of all definitions
      static constexpr UnitDefinition all_units[] = { m, mm, um, nm, ang, kg, g, Da, h, s, us, ns, ps, fs, C, ec, K, T, mol, particle, cd, J, eV, cal, kcal, rad, degree, no_unity, unknown };
      static constexpr int number_of_units = sizeof(all) / sizeof(UnitDefinition&);
    };

  }

}

#include "exanb/internal_units.h"

namespace exanb
{

  struct UnitSystem
  {
    EnumUnities m_length              = exanb::INTERNAL_UNIT_LENGTH;
    EnumUnities m_mass                = exanb::INTERNAL_UNIT_MASS;
    EnumUnities m_time                = exanb::INTERNAL_UNIT_TIME;
    EnumUnities m_electric_current    = exanb::INTERNAL_UNIT_ELECTRIC_CURRENT;
    EnumUnities m_temperature         = exanb::INTERNAL_UNIT_TEMPERATURE;
    EnumUnities m_amount_of_substance = exanb::INTERNAL_UNIT_AMOUNT_OF_SUBSTANCE;
    EnumUnities m_luminous_intensity  = exanb::INTERNAL_UNIT_LUMINOUS_INTENSITY;
    EnumUnities m_angle               = exanb::INTERNAL_UNIT_ANGLE;
  };

}


