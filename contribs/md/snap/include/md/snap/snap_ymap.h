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

namespace md
{
# include <md/snap/ymap_jmax2.hxx>
# include <md/snap/ymap_jmax3.hxx>
# include <md/snap/ymap_jmax4.hxx>

  static inline constexpr int snap_force_Y_count(int twojmax, int idxu_max)
  {
    const int jmax = twojmax / 2;
    if( jmax == 2 ) return Y_jmax2_jju_count;
    else if( jmax == 3 ) return Y_jmax3_jju_count;
    else if( jmax == 4 ) return Y_jmax4_jju_count;
    else return idxu_max;
  }
  
  static inline constexpr bool snap_force_use_Y(int twojmax , int jju)
  {
    const int jmax = twojmax / 2;
    if( jmax == 2 ) return Y_jmax2_jju_map[jju] != -1;
    else if( jmax == 3 ) return Y_jmax3_jju_map[jju] != -1;
    else if( jmax == 4 ) return Y_jmax4_jju_map[jju] != -1;
    else return true;
  }

  static inline constexpr int snap_force_Y_map(int twojmax , int jju)
  {
    const int jmax = twojmax / 2;
    if( jmax == 2 ) return Y_jmax2_jju_map[jju];
    else if( jmax == 3 ) return Y_jmax3_jju_map[jju];
    else if( jmax == 4 ) return Y_jmax4_jju_map[jju];
    else return jju;
  }

}
