#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <iterator> //std::begin std::end
#include <algorithm> //std::find_if
#include <assert.h>

namespace exanb
{

  // Use to create map between std string and enum type
  // Allow simple conversion between enumType and string and vice versa
  // the last element of enumType must be "unknow"
  // the size of s_keys have to be the same than the number of elements in EnumType
  template<class EnumType>
  class EnumMapHelper
  {
  public:

    EnumMapHelper(const std::vector<std::string>& keys)
      : m_keys(keys)
    {
      initMap();
    }

    void initMap()
    {
      //p_keys and the enum must have the same size 
      assert(static_cast<EnumType>(m_keys.size()-1) == EnumType::unknow);

      for(size_t i=0; i<m_keys.size(); ++i)
      {
        m_map.emplace(m_keys[i], static_cast<EnumType>(i));
      }
    }

    const std::unordered_map<std::string, EnumType>& getMap() const
    {
      return m_map;
    }

    EnumType fromString(const std::string& p_str)
    {
      //map must be initialize
      assert(m_map.size()!=0);

      return m_map.at(p_str);
    }

    std::string toString(const EnumType& p_enum)
    {
      //map must be initialize
      assert(m_map.size()!=0);

      auto it = std::find_if(std::begin(m_map), std::end(m_map),
                             [&](std::pair<std::string, EnumType> p) { return p.second == p_enum; });

      if (it == std::end(m_map))
        return m_keys.at(m_keys.size()-1); //last value is unknow

      return it->first;
    }

  private:
    std::vector<std::string> m_keys;
    std::unordered_map<std::string, EnumType> m_map;
  };

}


