#pragma once

#include <exanb/core/basic_types_def.h>
#include <algorithm>
#include <cmath>
#include <array>

namespace exanb
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

}

