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

#include <exanb/core/quaternion.h>
#include <exanb/core/basic_types_def.h>

#include <onika/cuda/cuda.h>

namespace exanb
{

  ONIKA_HOST_DEVICE_FUNC inline Mat3d quaternion_to_matrix(const Quaternion& q)
  {
    Mat3d m;
    m.m11 = q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z;
    m.m22 = q.w*q.w - q.x*q.x + q.y*q.y - q.z*q.z;
    m.m33 = q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z;
    m.m21 = 2.0 * (q.x*q.y + q.w*q.z ); 
    m.m12 = 2.0 * (q.x*q.y - q.w*q.z );
    m.m31 = 2.0 * (q.x*q.z - q.w*q.y );
    m.m13 = 2.0 * (q.x*q.z + q.w*q.y );
    m.m32 = 2.0 * (q.y*q.z + q.w*q.x );
    m.m23 = 2.0 * (q.y*q.z - q.w*q.x );
    return m;
  }

}

