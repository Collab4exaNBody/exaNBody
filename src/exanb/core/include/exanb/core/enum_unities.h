#pragma once

#include <vector>

namespace exanb
{

  enum class EnumUnities
  {
   //length
   meter,
   millimeter,
   micron,
   nanometer,
   angstrom,

   //mass
   kilogram,
   gram,
   atomic_mass_unit,

   //time
   hour,
   minute,
   second,
   microsecond,
   nanosecond,
   picosecond,
   fetosecond,

   //Electric current
   coulomb,
   elementary_charge,

   //temperature
   kelvin,
   degreeT,

   //amount of substance
   mol,
   particle,

   //luminous intensity
   candela,

   //others
   joule,
   electronVolt,
   calorie,
   kcalorie,
   radian,
   degreeA,

   // unitless
   no_unity,

   unknow
  };

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

  struct UnityWithPower
  {
    EnumUnities m_unity = EnumUnities::no_unity;
    int m_power = 0;
    inline bool operator == (const UnityWithPower& rhs) const { return m_unity==rhs.m_unity && m_power==rhs.m_power; }
  };

  using UnitiesDescriptor = std::vector< UnityWithPower >;

}


