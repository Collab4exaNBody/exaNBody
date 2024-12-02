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

#include <onika/math/basic_types.h>
#include <cmath>

namespace exanb
{

  struct Deformation
  {
    Vec3d m_angles = { M_PI/2, M_PI/2 , M_PI/2 };
    Vec3d m_extension = { 1. , 1. , 1. };
  };

  struct DeformationRange
  {
    Deformation m_min;
    Deformation m_max;
  };

}

