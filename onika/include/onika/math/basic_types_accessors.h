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

#include <onika/math/basic_types_def.h>
#include <algorithm>
#include <cmath>
#include <array>

namespace onika { namespace math
{
  // ================== accessors =======================
  
  inline Vec3d column1(const Mat3d& m) { return Vec3d{ m.m11, m.m21, m.m31 }; }
  inline Vec3d column2(const Mat3d& m) { return Vec3d{ m.m12, m.m22, m.m32 }; }
  inline Vec3d column3(const Mat3d& m) { return Vec3d{ m.m13, m.m23, m.m33 }; }
  inline Vec3d line1(const Mat3d& m) { return Vec3d{ m.m11, m.m12, m.m13 }; }
  inline Vec3d line2(const Mat3d& m) { return Vec3d{ m.m21, m.m22, m.m23 }; }
  inline Vec3d line3(const Mat3d& m) { return Vec3d{ m.m31, m.m32, m.m33 }; }
  inline void set_line1(Mat3d& m, const Vec3d& v ) { m.m11=v.x; m.m12=v.y; m.m13=v.z; }
  inline void set_line2(Mat3d& m, const Vec3d& v ) { m.m21=v.x; m.m22=v.y; m.m23=v.z; }
  inline void set_line3(Mat3d& m, const Vec3d& v ) { m.m31=v.x; m.m32=v.y; m.m33=v.z; }

  inline Vec3d diagonal(const Mat3d& mat) { return Vec3d{ mat.m11, mat.m22, mat.m33 }; }
  inline void set_diagonal(Mat3d& m, const Vec3d& v) { m.m11=v.x; m.m22=v.y; m.m33=v.z; }

} }

