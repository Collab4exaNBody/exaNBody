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

#include <type_traits>
#include <cstdint>
#include <cmath>
#include <exanb/core/basic_types_operators.h>

namespace exanb
{

  // ---- optional transform ------
  struct LinearXForm
  {
    Mat3d m_matrix;
    ONIKA_HOST_DEVICE_FUNC inline Vec3d transformCoord(const Vec3d& x) const noexcept { return m_matrix * x; }
  };

  struct NullXForm
  {
    ONIKA_HOST_DEVICE_FUNC static inline constexpr Vec3d transformCoord(Vec3d x) noexcept { return x; }
  };

}

