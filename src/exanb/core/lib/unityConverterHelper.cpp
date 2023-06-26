#include <exanb/core/unityConverterHelper.h>
#include <exanb/core/enumMapHelper.h>
#include <exanb/core/physics_constants.h>

#include <sstream>

namespace exanb
{

  EnumMapHelper<EnumUnities> UnityConverterHelper::s_enum_helper( {
      "m",
      "mm",
      "um",
      "nm",
      "ang",

      "kg",
      "g",
      "Da",

      "h",
      "m",
      "s",
      "us",
      "ns",
      "ps",
      "fs",

      "C",
      "e-",

      "K",
      "T",

      "mol",
      "particle",

      "cd",

      "J",
      "eV",
      "cal",
      "kcal",

      "rad",
      "degree",

      "1",//Add to be convenient. Example : unity = 1/ang

      "unknow"
      } );

  UnityConverterHelper::UnityConverterHelper()
  {
    m_powers[0]=0; // Length
    m_powers[1]=0; // Mass
    m_powers[2]=0; // Time
    m_powers[3]=0; // Electric current
    m_powers[4]=0; // Temperature
    m_powers[5]=0; // Amount of substance
    m_powers[6]=0; // Luminous intensity
    m_powers[7]=0; // Angle
  }

  UnityConverterHelper::UnityConverterHelper(int L, int M, int T, int I, int Theta, int N, int J, int A)
  {
    m_powers[0]=L; //length
    m_powers[1]=M; //mass
    m_powers[2]=T; //time
    m_powers[3]=I; //Electric current
    m_powers[4]=Theta; //Temperature
    m_powers[5]=N; // Amount of substance
    m_powers[6]=J; // Luminous intensity
    m_powers[7]=A; // Angle
  }

  /// @brief Copy constructor
  /// @param [in] quantity Another quantity
  UnityConverterHelper::UnityConverterHelper(const UnityConverterHelper& quantity) : m_powers(quantity.m_powers) 
  {
  }


  /// @brief Assignment operator
  /// @param [in] quantity Quantity to copy
  inline UnityConverterHelper& UnityConverterHelper::operator = (const UnityConverterHelper& quantity) {
    m_powers = quantity.m_powers;
    return *this;
  }


  /// @brief Multiplication operator
  ///
  /// Add the m_powers on each basic dimension
  /// @param [in] quantity Dimension to multiply to
  UnityConverterHelper UnityConverterHelper::operator * (const UnityConverterHelper& quantity) const {
    UnityConverterHelper tmp;
    std::transform(quantity.m_powers.begin(), quantity.m_powers.end(), this->m_powers.begin(), tmp.m_powers.begin(), std::plus<int>());
    return tmp;
  }


  /// @brief Division operator
  ///
  /// Subtract the m_powers on each basic dimension
  /// @param [in] quantity Dimension to divide by
  UnityConverterHelper UnityConverterHelper::operator / (const UnityConverterHelper& quantity) const {
    UnityConverterHelper tmp;
    std::transform(quantity.m_powers.begin(), quantity.m_powers.end(), this->m_powers.begin(), tmp.m_powers.begin(), std::minus<int>());
    return tmp;
  }


  /// @brief Equality operator
  /// @param [in] quantity UnityConverterHelper to compare to
  bool UnityConverterHelper::operator == (const UnityConverterHelper& quantity) const {
    return (this->m_powers == quantity.m_powers);
  }


  /// @brief Inequality operator
  /// @param [in] quantity UnityConverterHelper to compare to
  bool UnityConverterHelper::operator != (const UnityConverterHelper& quantity) const {
    return !(*this==quantity);
  }

  void UnityConverterHelper::addLengthPower           (const int& p_add)
  {
    m_powers.at(0) += p_add;
  }

  void UnityConverterHelper::addMassPower             (const int& p_add)
  {
    m_powers.at(1) += p_add;
  }
  void UnityConverterHelper::addTimePower             (const int& p_add)
  {
    m_powers.at(2) += p_add;
  }
  void UnityConverterHelper::addElectricCurrentPower  (const int& p_add)
  {
    m_powers.at(3) += p_add;
  }
  void UnityConverterHelper::addTemperaturePower      (const int& p_add)
  {
    m_powers.at(4) += p_add;
  }
  void UnityConverterHelper::addAmountOfSubstancePower(const int& p_add)
  {
    m_powers.at(5) += p_add;
  }
  void UnityConverterHelper::addLuminousIntensityPower(const int& p_add)
  {
    m_powers.at(6) += p_add;
  }
  void UnityConverterHelper::addAnglePower(const int& p_add)
  {
    m_powers.at(7) += p_add;
  }

  //non SI : kg m^2 /s^2
  void UnityConverterHelper::addEnergyPower(const int& p_add)
  {
    m_powers.at(1) += p_add;
    m_powers.at(0) += 2*p_add;
    m_powers.at(2) -= 2*p_add;
  }

  int UnityConverterHelper::getLengthPower() const
  {
    return m_powers.at(0);
  }
  int UnityConverterHelper::getMassPower() const
  {
    return m_powers.at(1);
  }
  int UnityConverterHelper::getTimePower() const
  {
    return m_powers.at(2);
  }
  int UnityConverterHelper::getElectricCurrentPower() const
  {
    return m_powers.at(3);
  }
  int UnityConverterHelper::getTemperaturePower() const
  {
    return m_powers.at(4);
  }
  int UnityConverterHelper::getAmountOfSubstancePower() const
  {
    return m_powers.at(5);
  }
  int UnityConverterHelper::getLuminousIntensityPower() const
  {
    return m_powers.at(6);
  }
  int UnityConverterHelper::getAnglePower() const
  {
    return m_powers.at(7);
  }


  //-----------------------------------------STATIC METHODS-------------------------------------------------------

  void UnityConverterHelper::parse_unities_descriptor( std::string p_unities, UnitiesDescriptor& unities )
  {
    // Take in input unities
    // Split unities, and take the power associated
    // Example :
    // input s="ang^6/kcal/J/T^7/kg^30"
    // output unities_list = ({ang,6}, {kcal,-1}, {J,-1}, {T,-7}, {kg,-30})
    size_t pos = 0;
    int power = 1;//first unit positive
    std::vector< std::pair<std::string, int> > unities_list;
    while ((pos = p_unities.find_first_of("/*.")) != std::string::npos)
    {
      unities_list.push_back(std::make_pair(p_unities.substr(0, pos), power));
      ((p_unities.compare(pos, 1, "/"))==0)?power=-1:power=1;
      p_unities.erase(0, pos + 1);
    }
    unities_list.push_back(std::make_pair(p_unities.substr(0, pos), power));//to take into account the last unit

    int power_value = 0;
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
    
    unities.clear();
    
    for(const auto& pair : unities_list)
    {
      //Protection : if the unity doesn't exist, exit the code
      if(s_enum_helper.getMap().find(pair.first)==s_enum_helper.getMap().end())
	    {
	      std::cerr << "[WARNING] UnityConverterHelper::getValueSI : unity " << pair.first << " is not defined as a code unity." << std::endl;
	      std::cerr << "[WARNING] UnityConverterHelper::getValueSI : accepted unities are : ";
	      for(const auto& pair : s_enum_helper.getMap())
	      {
	        std::cerr << pair.first << " " ;
	      }
	      std::cerr << "." << std::endl;
	      std::abort();
	    }
	    unities.push_back( UnityWithPower{ s_enum_helper.getMap().at(pair.first) , pair.second } );
    }
    
  }

  std::string UnityConverterHelper::unities_descriptor_to_string( const UnitiesDescriptor& unities )
  {
    std::ostringstream oss;
    bool sep = false;
    for(UnityWithPower up : unities)
    {
      int p = up.m_power;
      if( sep )
      {
        if( p > 0 )
        {
          oss << '*';
        }
        else if( p < 0 )
        {
          oss << '/';
          p = -p;
        }
      }
      sep = true;
      if(p != 0)
      {
        oss << s_enum_helper.toString( up.m_unity ) ;
        if( p != 1 )
        {
          oss << '^' << p;
        }
      }
    }
    return oss.str();
  }

  UnityConverterHelper UnityConverterHelper::getValueSI(double& p_value, const UnitiesDescriptor& p_unities)
  {

    //make the conversion in order to have SI units
    UnityConverterHelper unities_in_SI;
    for(UnityWithPower unity_power : p_unities)
    {

      switch( unity_power.m_unity )
	    {
	      //length
	    case EnumUnities::meter:
	      unities_in_SI.addLengthPower(unity_power.m_power);
	      break;
	    case EnumUnities::millimeter:
	      p_value *= std::pow(1.0e-3, unity_power.m_power);
	      unities_in_SI.addLengthPower(unity_power.m_power);
	      break;
	    case EnumUnities::micron :
	      p_value *= std::pow(1.0e-6, unity_power.m_power);
	      unities_in_SI.addLengthPower(unity_power.m_power);
	      break;
	    case EnumUnities::nanometer :
	      p_value *= std::pow(1.0e-9, unity_power.m_power);
	      unities_in_SI.addLengthPower(unity_power.m_power);
	      break;
	    case EnumUnities::angstrom :
	      p_value *= std::pow(1.0e-10, unity_power.m_power);
	      unities_in_SI.addLengthPower(unity_power.m_power);
	      break;

	      //mass
	    case EnumUnities::kilogram :
	      unities_in_SI.addMassPower(unity_power.m_power);
	      break;
	    case EnumUnities::gram :
	      p_value *= std::pow(1.0e-3, unity_power.m_power);
	      unities_in_SI.addMassPower(unity_power.m_power);
	      break;
	    case EnumUnities::atomic_mass_unit :
	      p_value *= std::pow( 1.660539040e-27, unity_power.m_power);
	      unities_in_SI.addMassPower(unity_power.m_power);
	      break;

	      //time
	    case EnumUnities::hour :
	      p_value *= std::pow(3600, unity_power.m_power);
	      unities_in_SI.addTimePower(unity_power.m_power);
	      break;
	    case EnumUnities::minute :
	      p_value *= std::pow(60, unity_power.m_power);
	      unities_in_SI.addTimePower(unity_power.m_power);
	      break;
	    case EnumUnities::second :
	      unities_in_SI.addTimePower(unity_power.m_power);
	      break;
	    case EnumUnities::microsecond :
	      p_value *= std::pow(1.0e-6, unity_power.m_power);
	      unities_in_SI.addTimePower(unity_power.m_power);
	      break;
	    case EnumUnities::nanosecond :
	      p_value *= std::pow(1.0e-9, unity_power.m_power);
	      unities_in_SI.addTimePower(unity_power.m_power);
	      break;
	    case EnumUnities::picosecond :
	      p_value *= std::pow(1.0e-12, unity_power.m_power);
	      unities_in_SI.addTimePower(unity_power.m_power);
	      break;
	    case EnumUnities::fetosecond :
	      p_value *= std::pow(1.0e-15, unity_power.m_power);
	      unities_in_SI.addTimePower(unity_power.m_power);
	      break;

	      //Electric current
	    case EnumUnities::coulomb :
	      unities_in_SI.addElectricCurrentPower(unity_power.m_power);
	      break;
	    case EnumUnities::elementary_charge :
          //legacy_constant::elementaryCharge = 1.6021892e-19
          // This is the internal unit for charge
	      p_value *= std::pow(legacy_constant::elementaryCharge , unity_power.m_power);
	      unities_in_SI.addElectricCurrentPower(unity_power.m_power);
	      break;

	      //temperature
	    case EnumUnities::kelvin :
	      unities_in_SI.addTemperaturePower(unity_power.m_power);
	      break;
	      //case EnumUnities::degreeT :
	      // Dangerous : can have division by 0
	      //break;

	      //amount of substance
	    case EnumUnities::mol :
	      unities_in_SI.addAmountOfSubstancePower(unity_power.m_power);
	      break;
        case EnumUnities::particle:
          p_value *= std::pow( (1/6.02214076)*1.0e23 , unity_power.m_power);
          unities_in_SI.addAmountOfSubstancePower(unity_power.m_power);

	      //luminous intensity
	    case EnumUnities::candela :
	      unities_in_SI.addLuminousIntensityPower(unity_power.m_power);
	      break;

	      //others
	    case EnumUnities::joule :
	      unities_in_SI.addEnergyPower(unity_power.m_power);
	      break;
	    case EnumUnities::electronVolt :
	      p_value *= std::pow( legacy_constant::elementaryCharge , unity_power.m_power );
	      unities_in_SI.addEnergyPower(unity_power.m_power);
	      break;
	    case EnumUnities::calorie :
	      p_value *= std::pow(4.1868, unity_power.m_power);
	      unities_in_SI.addEnergyPower(unity_power.m_power);
	      break;
	    case EnumUnities::kcalorie :
	      p_value *= std::pow(4186.8, unity_power.m_power);
	      unities_in_SI.addEnergyPower(unity_power.m_power);
	      break;
	    case EnumUnities::radian :
	      break;
	    case EnumUnities::degreeA :
	      p_value *= std::pow(M_PI/180.0, unity_power.m_power);
	      break;
	    default:
	      break;
	    }
    }
    
    return unities_in_SI;
  }

  double UnityConverterHelper::convert(const YAML::Node& p_node)
  {
    double value  = p_node["value"].as<double>();
    std::string s = p_node["unity"].as<std::string>();
    UnitiesDescriptor unities;
    parse_unities_descriptor( s, unities );
    return convert(value, unities);
  }

  double UnityConverterHelper::	convert(
               double p_value,
               const std::string& str_unities,
				       EnumUnities p_length,
				       EnumUnities p_mass,
				       EnumUnities p_time,
				       EnumUnities p_electric_current,
				       EnumUnities p_temperature,
				       EnumUnities p_amount_of_substance,
				       EnumUnities p_luminous_intensity,
				       EnumUnities p_angle)
  {
    UnitiesDescriptor unities;
    parse_unities_descriptor( str_unities, unities );    
    return UnityConverterHelper::convert( p_value, unities, p_length, p_mass, p_time, p_electric_current, p_temperature, p_amount_of_substance, p_luminous_intensity, p_angle);
  }

  double UnityConverterHelper::convert(double p_value,
                                       const UnitiesDescriptor& p_unities_with_powers_in,
                                       EnumUnities p_length,
                                       EnumUnities p_mass,
                                       EnumUnities p_time,
                                       EnumUnities p_electric_current,
                                       EnumUnities p_temperature,
                                       EnumUnities p_amount_of_substance,
                                       EnumUnities p_luminous_intensity,
                                       EnumUnities p_angle)
  {

    // filters out units that are the same as output units top avoid rounding errors
    UnitiesDescriptor p_unities_with_powers;
    for(const auto& U : p_unities_with_powers_in )
    {
      if( U.m_unity!=p_length
       && U.m_unity!=p_mass
       && U.m_unity!=p_time
       && U.m_unity!=p_electric_current
       && U.m_unity!=p_temperature
       && U.m_unity!=p_amount_of_substance
       && U.m_unity!=p_luminous_intensity
       && U.m_unity!=p_angle )
       p_unities_with_powers.push_back( U );
    } 
    // std::cout << "p_unities_with_powers_in = " << unities_descriptor_to_string(p_unities_with_powers_in) << std::endl;
    // std::cout << "p_unities_with_powers = " << unities_descriptor_to_string(p_unities_with_powers) << std::endl;

    UnityConverterHelper unities = getValueSI(p_value, p_unities_with_powers);

    //length
    switch(p_length)
      {
      case EnumUnities::meter:
        break;
      case EnumUnities::millimeter:
        p_value *= std::pow(1.0e3, unities.getLengthPower());
        break;
      case EnumUnities::micron :
        p_value *= std::pow(1.0e6, unities.getLengthPower());
        break;
      case EnumUnities::nanometer :
        p_value *= std::pow(1.0e9, unities.getLengthPower());
        break;
      case EnumUnities::angstrom :
        p_value *= std::pow(1.0e10, unities.getLengthPower());
        break;
      default:
        std::cerr << "[WARNING] UnityConverterHelper::convert : length unity " << s_enum_helper.toString(p_length) 
		  << " unknown !" << std::endl;
        std::abort();
      }


    //mass  
    switch(p_mass)
      {
      case EnumUnities::kilogram :
        break;
      case EnumUnities::gram :
        p_value *= std::pow(1.0e3, unities.getMassPower());
        break;
      case EnumUnities::atomic_mass_unit :
        // Th.C. : equivalent but produces exact same values as in ExaStamp v1.x
        p_value *= std::pow( 6.0221408578238376985e+26 /*1./1.6605e-27*/, unities.getMassPower());
        break;
      default:
        std::cerr << "[WARNING] UnityConverterHelper::convert : mass unity " << s_enum_helper.toString(p_mass) 
                  << " unknown !" << std::endl;
        std::abort();
      }

    //time 
    switch(p_time)
      {
      case EnumUnities::hour :
        p_value *= std::pow(1.0/3600, unities.getTimePower());
        break;
      case EnumUnities::minute :
        p_value *= std::pow(1.0/60, unities.getTimePower());
        break;
      case EnumUnities::second :
        break;
      case EnumUnities::microsecond :
        p_value *= std::pow(1.0e6, unities.getTimePower());
        break;
      case EnumUnities::nanosecond :
        p_value *= std::pow(1.0e9, unities.getTimePower());
        break;
      case EnumUnities::picosecond :
        p_value *= std::pow(1.0e12, unities.getTimePower());
        break;
      case EnumUnities::fetosecond :
        p_value *= std::pow(1.0e15, unities.getTimePower());
        break;
      default:
        std::cerr << "[WARNING] UnityConverterHelper::convert : time unity " << s_enum_helper.toString(p_time) 
		  << " unknown !" << std::endl;
        std::abort();
      }

    //Electric current
    switch(p_electric_current)
      {
      case EnumUnities::coulomb :
        break;
      case EnumUnities::elementary_charge :
        p_value *= std::pow(1./legacy_constant::elementaryCharge, unities.getElectricCurrentPower());
        break;
      default:
        std::cerr << "[WARNING] UnityConverterHelper::convert : electric current unity " << s_enum_helper.toString(p_electric_current) 
		  << " unknown !" << std::endl;
        std::abort();
      }

    //temperature
    switch(p_temperature)
      {
	  case EnumUnities::kelvin :
	    break;
	    //case EnumUnities::degreeT :
	    // Dangerous : can have division by 0
	    //break;
      default:
        std::cerr << "[WARNING] UnityConverterHelper::convert : temperature unity " << s_enum_helper.toString(p_temperature) 
		  << " unknown !" << std::endl;
        std::abort();
      }

    //amount of substance
    switch(p_amount_of_substance)
      {
      case EnumUnities::mol :
        break;
      case EnumUnities::particle :
        p_value *= std::pow(6.02214076e23 , unities.getAmountOfSubstancePower());
        break;
      default:
        std::cerr << "[WARNING] UnityConverterHelper::convert : amount of substance unity " << s_enum_helper.toString(p_amount_of_substance) 
		  << " unknown !" << std::endl;
        std::abort();
      }

    //luminous intensity
    switch(p_luminous_intensity)
      {
      case EnumUnities::candela :
        break;
      default:
        std::cerr << "[WARNING] UnityConverterHelper::convert : luminous intensity unity " << s_enum_helper.toString(p_luminous_intensity) 
		  << " unknown !" << std::endl;
        std::abort();
      }

    //Angle
    switch(p_angle)
      {
      case EnumUnities::radian :
        break;
      case EnumUnities::degreeA :
        p_value *= std::pow(180.0/M_PI, unities.getAnglePower());
        break;
      default:
        std::cerr << "[WARNING] UnityConverterHelper::convert : angle unity " << s_enum_helper.toString(p_angle) 
		  << " unknown !" << std::endl;
        std::abort();
      }

    return p_value;
  }



  /// @brief Print to debug
  void UnityConverterHelper::debug() const 
  {
    std::cout <<std::endl;
    std::cout << "[DEBUG] UnityConverterHelper class use to convert unities" << std::endl;
    std::cout << "[DEBUG] length power : "              << getLengthPower()            << std::endl;
    std::cout << "[DEBUG] mass power : "                << getMassPower()              << std::endl;
    std::cout << "[DEBUG] time power : "                << getTimePower()              << std::endl;
    std::cout << "[DEBUG] electric current power : "    << getElectricCurrentPower()   << std::endl;
    std::cout << "[DEBUG] temperature power : "         << getTemperaturePower()       << std::endl;
    std::cout << "[DEBUG] amount of substance power : " << getAmountOfSubstancePower() << std::endl;
    std::cout << "[DEBUG] luminous intensity power : "  << getLuminousIntensityPower() << std::endl;
    std::cout << "[DEBUG] angle power : "               << getAnglePower()             << std::endl;
    std::cout <<std::endl;
  }
  
} // end of namespace exanb


