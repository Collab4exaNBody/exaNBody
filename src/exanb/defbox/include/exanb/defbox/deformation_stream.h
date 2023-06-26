#pragma once

#include <exanb/defbox/deformation.h>
#include <exanb/core/basic_types_stream.h>
#include <iostream>

namespace exanb
{

  inline std::ostream& operator << (std::ostream& out, const Deformation& defbox)
  {
    out<<"{ angles="<<defbox.m_angles<<" , extension="<<defbox.m_extension<<" }";
    return out;
  }

}
