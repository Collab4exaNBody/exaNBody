#pragma once

#include <iostream>
#include <array>
#include <sstream>
#include <vector>
#include <utility>
#include <cmath>

#include <yaml-cpp/yaml.h>

#include <exanb/core/enumMapHelper.h>
#include <exanb/core/enum_unities.h>

namespace exanb
{

  /// @brief Represents the dimension of a physical quantity
  ///
  /// The dimension of a physical quantity can be expressed as a product
  /// of the basic physical dimensions raised to a rational power
  class UnityConverterHelper {

  public:

    static void parse_unities_descriptor( std::string s, UnitiesDescriptor& unities );
    static std::string unities_descriptor_to_string( const UnitiesDescriptor& unities );

    static double convert(const YAML::Node& p_node);
    static double convert(double p_value,
		          const UnitiesDescriptor& p_unities,
              EnumUnities p_length              = exanb::INTERNAL_UNIT_LENGTH,
              EnumUnities p_mass                = exanb::INTERNAL_UNIT_MASS,
              EnumUnities p_time                = exanb::INTERNAL_UNIT_TIME,
              EnumUnities p_electric_current    = exanb::INTERNAL_UNIT_ELECTRIC_CURRENT,
              EnumUnities p_temperature         = exanb::INTERNAL_UNIT_TEMPERATURE,
              EnumUnities p_amount_of_substance = exanb::INTERNAL_UNIT_AMOUNT_OF_SUBSTANCE,
              EnumUnities p_luminous_intensity  = exanb::INTERNAL_UNIT_LUMINOUS_INTENSITY,
              EnumUnities p_angle               = exanb::INTERNAL_UNIT_ANGLE
              );

    static double convert(double p_value,
		          const std::string& p_unities,
              EnumUnities p_length              = exanb::INTERNAL_UNIT_LENGTH,
              EnumUnities p_mass                = exanb::INTERNAL_UNIT_MASS,
              EnumUnities p_time                = exanb::INTERNAL_UNIT_TIME,
              EnumUnities p_electric_current    = exanb::INTERNAL_UNIT_ELECTRIC_CURRENT,
              EnumUnities p_temperature         = exanb::INTERNAL_UNIT_TEMPERATURE,
              EnumUnities p_amount_of_substance = exanb::INTERNAL_UNIT_AMOUNT_OF_SUBSTANCE,
              EnumUnities p_luminous_intensity  = exanb::INTERNAL_UNIT_LUMINOUS_INTENSITY,
              EnumUnities p_angle               = exanb::INTERNAL_UNIT_ANGLE
              );


    void debug() const;

  private :
    UnityConverterHelper();
    UnityConverterHelper(int L, int M, int T, int I, int Theta, int N, int J, int A);
    UnityConverterHelper(const UnityConverterHelper& quantity);

    UnityConverterHelper& operator = (const UnityConverterHelper& quantity);

    UnityConverterHelper operator * (const UnityConverterHelper& quantity) const;
    UnityConverterHelper operator / (const UnityConverterHelper& quantity) const;

    bool operator == (const UnityConverterHelper& quantity) const;
    bool operator != (const UnityConverterHelper& quantity) const;

    void addLengthPower           (const int&);
    void addMassPower             (const int&);
    void addTimePower             (const int&);
    void addElectricCurrentPower  (const int&);
    void addTemperaturePower      (const int&);
    void addAmountOfSubstancePower(const int&);
    void addLuminousIntensityPower(const int&);
    void addAnglePower            (const int&);

    int getLengthPower           () const;
    int getMassPower             () const;
    int getTimePower             () const;
    int getElectricCurrentPower  () const;
    int getTemperaturePower      () const;
    int getAmountOfSubstancePower() const;
    int getLuminousIntensityPower() const;
    int getAnglePower            () const;

    //kg m2/s2
    void addEnergyPower(const int&);

    std::array<int,8> m_powers; ///< Vector to store the powers to apply to the basic physical dimensions to get this dimension

    static UnityConverterHelper getValueSI(double&,const UnitiesDescriptor&);


    static EnumMapHelper<EnumUnities> s_enum_helper;
  };

}

