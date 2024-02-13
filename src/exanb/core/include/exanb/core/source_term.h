#pragma once

#include <exanb/core/basic_types.h>
#include <vector>
#include <string>
#include <memory>

#include <yaml-cpp/yaml.h>

namespace exanb
{

  struct ScalarSourceTerm
  {
    virtual inline double operator () ( exanb::Vec3d r, double t=0.0, int64_t id=-1 ) const { return 0.0; }
    virtual inline ~ScalarSourceTerm() = default;
  };

  using ScalarSourceTermInstance = std::shared_ptr<ScalarSourceTerm>;

  ScalarSourceTermInstance make_source_term( const YAML::Node& node );
}

namespace YAML
{

  template<> struct convert< exanb::ScalarSourceTermInstance >
  {
    static inline bool decode(const Node& node, exanb::ScalarSourceTermInstance& source_term_func )
    {
      auto f = exanb::make_source_term( node );
      if( f != nullptr )
      {
        source_term_func = f;
        return true;
      }
      else
      {
        return false;
      }
    }
  };
  
}
