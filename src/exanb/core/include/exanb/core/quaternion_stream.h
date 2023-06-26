#pragma once

#include <exanb/core/quaternion.h>
#include <iostream>

namespace exanb
{

  inline std::ostream& operator << (std::ostream& out, const Quaternion& q)
  {
    out<<'('<<q.x<<','<<q.y<<','<<q.z<<','<<q.w<<')';
    return out;
  }

}

