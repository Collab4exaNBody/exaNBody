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

#include <exanb/core/basic_types_def.h>
#include <exanb/core/basic_types_constructors.h>
#include <algorithm>
#include <cmath>
#include <array>
#include <functional>
#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_math.h>
// #include <bit>

namespace exanb
{

  // =================== operations ===================

  ONIKA_HOST_DEVICE_FUNC inline bool operator < (const IJK& u, const IJK& v)
  {
    if(u.k<v.k) return true;
    if(u.k>v.k) return false;
    if(u.j<v.j) return true;
    if(u.j>v.j) return false;
    if(u.i<v.i) return true;
    return false;
  }

  ONIKA_HOST_DEVICE_FUNC inline double complex_norm (const Complexd& c)
  {
   return c.r*c.r + c.i*c.i;
  }

  ONIKA_HOST_DEVICE_FUNC inline Complexd operator * (const Complexd& c , double q)
  {
   return { c.r * q , c.i * q };
  }
  
  ONIKA_HOST_DEVICE_FUNC inline Complexd operator * (double q , const Complexd& c)
  {
   return { c.r * q , c.i * q };
  }

  ONIKA_HOST_DEVICE_FUNC inline Complexd& operator += (Complexd& a , const Complexd& b)
  {
    a.r += b.r;
    a.i += b.i;
    return a;
  }

  ONIKA_HOST_DEVICE_FUNC inline Vec3d min (const Vec3d& a, const Vec3d& b)
  {
    return Vec3d{ onika::cuda::min(a.x,b.x), onika::cuda::min(a.y,b.y), onika::cuda::min(a.z,b.z) };
  }

  ONIKA_HOST_DEVICE_FUNC inline Vec3d max(const Vec3d& a, const Vec3d& b)
  {
    return Vec3d{ onika::cuda::max(a.x,b.x), onika::cuda::max(a.y,b.y), onika::cuda::max(a.z,b.z) };
  }

  inline Vec3d floor(const Vec3d& a)
  {
    return Vec3d{ std::floor(a.x), std::floor(a.y), std::floor(a.z) };
  }

  inline Vec3d ceil(const Vec3d& a)
  {
    return Vec3d{ std::ceil(a.x), std::ceil(a.y), std::ceil(a.z) };
  }

  ONIKA_HOST_DEVICE_FUNC inline Vec3d operator - (Vec3d& a)
  {
    return Vec3d{ -a.x, -a.y, -a.z };
  }

  inline Vec3d vabs(const Vec3d& a)
  {
    return Vec3d{ std::fabs(a.x), std::fabs(a.y), std::fabs(a.z) };
  }

  ONIKA_HOST_DEVICE_FUNC inline Vec3d operator - (const Vec3d& a)
  {
    return Vec3d{ -a.x, -a.y, -a.z };
  }
  
  ONIKA_HOST_DEVICE_FUNC inline Vec3d operator - (const Vec3d& a, const Vec3d& b)
  {
    return Vec3d{ a.x-b.x, a.y-b.y, a.z-b.z };
  }

  ONIKA_HOST_DEVICE_FUNC inline Vec3d operator + (const Vec3d& a, const Vec3d& b)
  {
    return Vec3d{ a.x+b.x, a.y+b.y, a.z+b.z };
  }

  ONIKA_HOST_DEVICE_FUNC inline Vec3d operator * (const Vec3d& a, const Vec3d& b)
  {
    return Vec3d{ a.x*b.x, a.y*b.y, a.z*b.z };
  }

  ONIKA_HOST_DEVICE_FUNC inline Vec3d operator / (const Vec3d& a, const Vec3d& b)
  {
    return Vec3d{ a.x/b.x, a.y/b.y, a.z/b.z };
  }

  ONIKA_HOST_DEVICE_FUNC inline Vec3d operator * (const Vec3d& p, double scale)
  {
    return Vec3d{ p.x*scale, p.y*scale, p.z*scale };
  }
  ONIKA_HOST_DEVICE_FUNC inline Vec3d operator * (double scale, const Vec3d& p)
  {
    return Vec3d{ p.x*scale, p.y*scale, p.z*scale };
  }

  ONIKA_HOST_DEVICE_FUNC inline Vec3d operator + (const Vec3d& a, const IJK& b)
  {
    return Vec3d{ a.x+b.i, a.y+b.j, a.z+b.k };
  }

  ONIKA_HOST_DEVICE_FUNC inline Vec3d operator * (const Vec3d& a, const IJK& b)
  {
    return Vec3d{ a.x*b.i, a.y*b.j, a.z*b.k };
  }

  ONIKA_HOST_DEVICE_FUNC inline Vec3d operator / (const Vec3d& a, const IJK& b)
  {
    return Vec3d{ a.x/b.i, a.y/b.j, a.z/b.k };
  }

  ONIKA_HOST_DEVICE_FUNC inline Vec3d& operator /= (Vec3d& a, double b)
  {
    a.x /= b;
    a.y /= b;
    a.z /= b;
    return a;
  }

  ONIKA_HOST_DEVICE_FUNC inline IJK vclamp(const IJK& a, const IJK& min, const IJK& max)
  {
    using onika::cuda::clamp;
    return IJK{ clamp(a.i,min.i,max.i) , clamp(a.j,min.j,max.j) , clamp(a.k,min.k,max.k) };
  }

  ONIKA_HOST_DEVICE_FUNC inline IJK vclamp(const IJK& a, ssize_t min, ssize_t max)
  {
    using onika::cuda::clamp;
    return IJK{ clamp(a.i,min,max) , clamp(a.j,min,max) , clamp(a.k,min,max) };
  }

  ONIKA_HOST_DEVICE_FUNC inline IJK operator + (const IJK& a, const IJK& b)
  {
    return IJK{ a.i+b.i, a.j+b.j, a.k+b.k };
  }

  ONIKA_HOST_DEVICE_FUNC inline IJK operator - (const IJK& a, const IJK& b)
  {
    return IJK{ a.i-b.i, a.j-b.j, a.k-b.k };
  }

  ONIKA_HOST_DEVICE_FUNC inline IJK operator * (const IJK& a, const IJK& b)
  {
    return IJK{ a.i*b.i, a.j*b.j, a.k*b.k };
  }

  ONIKA_HOST_DEVICE_FUNC inline IJK operator / (const IJK& a, const IJK& b)
  {
    return IJK{ a.i/b.i, a.j/b.j, a.k/b.k };
  }

  ONIKA_HOST_DEVICE_FUNC inline IJK operator % (const IJK& a, const IJK& b)
  {
    return IJK{ a.i%b.i, a.j%b.j, a.k%b.k };
  }

  ONIKA_HOST_DEVICE_FUNC inline bool operator == (const IJK& a, const IJK& b)
  {
    return a.i==b.i && a.j==b.j && a.k==b.k;
  }

  ONIKA_HOST_DEVICE_FUNC inline bool operator != (const IJK& a, const IJK& b)
  {
    return a.i!=b.i || a.j!=b.j || a.k!=b.k;
  }

  ONIKA_HOST_DEVICE_FUNC inline IJK operator + (const IJK& p, ssize_t a)
  {
    return IJK{ p.i+a, p.j+a, p.k+a };
  }

  ONIKA_HOST_DEVICE_FUNC inline IJK operator - (const IJK& p, ssize_t a)
  {
    return IJK{ p.i-a, p.j-a, p.k-a };
  }

  ONIKA_HOST_DEVICE_FUNC inline IJK operator * (const IJK& p, ssize_t a)
  {
    return IJK{ p.i*a, p.j*a, p.k*a };
  }

  ONIKA_HOST_DEVICE_FUNC inline IJK operator / (const IJK& p, ssize_t a)
  {
    return IJK{ p.i/a, p.j/a, p.k/a };
  }

  ONIKA_HOST_DEVICE_FUNC inline IJK operator % (const IJK& p, ssize_t a)
  {
    return IJK{ p.i%a, p.j%a, p.k%a };
  }

  ONIKA_HOST_DEVICE_FUNC inline IJK& operator += (IJK& a, const IJK& b)
  {
    a = a + b;
    return a;
  }

  ONIKA_HOST_DEVICE_FUNC inline constexpr bool lexicographic_order(IJK a, IJK b)
  {
    if(a.k<b.k) return true;
    if(a.k>b.k) return false;
    if(a.j<b.j) return true;
    if(a.j>b.j) return false;
    if(a.i<b.i) return true;
    return false;
  }

  ONIKA_HOST_DEVICE_FUNC inline GridBlock operator - (const GridBlock& b, const IJK& p)
  {
    return GridBlock{ b.start-p , b.end-p };
  }
  ONIKA_HOST_DEVICE_FUNC inline GridBlock operator + (const GridBlock& b, const IJK& p)
  {
    return GridBlock{ b.start+p , b.end+p };
  }

  ONIKA_HOST_DEVICE_FUNC inline Vec3d operator * (const IJK& p, double scale)
  {
    return Vec3d{ p.i*scale, p.j*scale, p.k*scale };
  }

  ONIKA_HOST_DEVICE_FUNC inline Vec3d operator / (const IJK& p, double div)
  {
    return Vec3d{ p.i/div, p.j/div, p.k/div };
  }

  ONIKA_HOST_DEVICE_FUNC inline Vec3d operator / (const Vec3d& p, double div)
  {
    return Vec3d{ p.x/div, p.y/div, p.z/div };
  }

  ONIKA_HOST_DEVICE_FUNC inline Vec3d operator + (const Vec3d& p, double a)
  {
    return Vec3d{ p.x+a, p.y+a, p.z+a };
  }

  ONIKA_HOST_DEVICE_FUNC inline double norm2 (const Vec3d& p)
  {
    return ( p.x * p.x ) + ( p.y * p.y ) + ( p.z * p.z );
  }

  ONIKA_HOST_DEVICE_FUNC inline double norm (const Vec3d& p)
  {
   return sqrt( norm2(p) );
  }

  ONIKA_HOST_DEVICE_FUNC inline Vec3d& operator += (Vec3d& a, const Vec3d& b)
  {
    a = a + b;
    return a;
  }

  ONIKA_HOST_DEVICE_FUNC inline Vec3d& operator -= (Vec3d& a, const Vec3d& b)
  {
    a = a - b;
    return a;
  }

  ONIKA_HOST_DEVICE_FUNC inline Vec3d& operator *= (Vec3d& a, const Vec3d& b)
  {
    a = a * b;
    return a;
  }

  ONIKA_HOST_DEVICE_FUNC inline Vec3d& operator *= (Vec3d& a, double b)
  {
    a = a * b;
    return a;
  }

  ONIKA_HOST_DEVICE_FUNC inline Vec3d operator - (const Vec3d& p, double a)
  {
    return Vec3d{ p.x-a, p.y-a, p.z-a };
  }

  ONIKA_HOST_DEVICE_FUNC inline bool operator == (const Vec3d& a, const Vec3d& b)
  {
    return a.x==b.x && a.y==b.y && a.z==b.z;
  }
  ONIKA_HOST_DEVICE_FUNC inline bool operator != (const Vec3d& a, const Vec3d& b)
  {
    return ! ( a == b );
  }

  ONIKA_HOST_DEVICE_FUNC inline Vec3d reciprocal(const Vec3d& a)
  {
    return Vec3d{1./a.x,1./a.y,1./a.z};
  }

  ONIKA_HOST_DEVICE_FUNC inline Vec3d cross(const Vec3d &u, const Vec3d &v) {
    return Vec3d{ u.y*v.z - u.z*v.y,
		              u.z*v.x - u.x*v.z,
		              u.x*v.y - u.y*v.x };    
  }

  ONIKA_HOST_DEVICE_FUNC inline double dot(const Vec3d &u, const Vec3d &v)
  {
    return u.x*v.x + u.y*v.y + u.z*v.z;
  }

  ONIKA_HOST_DEVICE_FUNC
  inline double angle(const Vec3d& a, const Vec3d& b)
  {
    //double err = 1.0e-4;
    double cosA = dot(a,b) / sqrt( dot(a,a)*dot(b,b) );
    //When angle is near 0 or pi, numerical error can give incoherent result
    if(cosA <= -1.0 ) { return M_PI; }
    else if(cosA >= 1.0) { return 0.0; }
    return acos(cosA);
  }

  ONIKA_HOST_DEVICE_FUNC inline bool operator == (const GridBlock& a, const GridBlock& b)
  {
    return a.start==b.start && a.end==b.end ;
  }

  ONIKA_HOST_DEVICE_FUNC inline bool operator == (const AABB& a, const AABB& b)
  {
    return a.bmin==b.bmin && a.bmax==b.bmax ;
  }

  ONIKA_HOST_DEVICE_FUNC inline bool operator != (const AABB& a, const AABB& b)
  {
    return a.bmin!=b.bmin || a.bmax!=b.bmax ;
  }

  ONIKA_HOST_DEVICE_FUNC inline bool operator != (const GridBlock& a, const GridBlock& b)
  {
    return ! ( a == b ) ;
  }


  /// @brief Matrix product 3x1 to 1x3 to produce 3x3 matrix
  /// @tparam T Type inside
  /// @param [in] u 3x1 matrix
  /// @param [in] v 1x3 matrix
  /// @return Result matrix
  ONIKA_HOST_DEVICE_FUNC inline Mat3d tensor( const Vec3d& u, const Vec3d& v)
  {
    return Mat3d{u.x*v.x, u.x*v.y, u.x*v.z, u.y*v.x, u.y*v.y, u.y*v.z, u.z*v.x, u.z*v.y, u.z*v.z};
  }
  
  ONIKA_HOST_DEVICE_FUNC inline Mat3d operator + ( const Mat3d& a, const Mat3d& b)
  {
    return Mat3d{
      a.m11+b.m11,
      a.m12+b.m12,
      a.m13+b.m13,

      a.m21+b.m21,
      a.m22+b.m22,
      a.m23+b.m23,

      a.m31+b.m31,
      a.m32+b.m32,
      a.m33+b.m33 };
  }

  ONIKA_HOST_DEVICE_FUNC inline Vec3d operator * ( const Mat3d& a, const Vec3d& u)
  {
    return Vec3d{ a.m11*u.x + a.m12*u.y + a.m13*u.z, a.m21*u.x + a.m22*u.y + a.m23*u.z, a.m31*u.x + a.m32*u.y + a.m33*u.z };
  }

  ONIKA_HOST_DEVICE_FUNC inline Vec3d operator * ( const Mat3d& a, const double (&u) [3])
  {
   return Vec3d{ a.m11*u[0] + a.m12*u[1] + a.m13*u[2], a.m21*u[0] + a.m22*u[1] + a.m23*u[2], a.m31*u[0] + a.m32*u[1] + a.m33*u[2] };
  }

  ONIKA_HOST_DEVICE_FUNC inline Mat3d operator - ( const Mat3d& a, const Mat3d& b)
  {
    return Mat3d{
      a.m11-b.m11,
      a.m12-b.m12,
      a.m13-b.m13,

      a.m21-b.m21,
      a.m22-b.m22,
      a.m23-b.m23,

      a.m31-b.m31,
      a.m32-b.m32,
      a.m33-b.m33 };
  }

  ONIKA_HOST_DEVICE_FUNC inline Mat3d& operator += ( Mat3d& a, const Mat3d& b)
  {
    a = a + b;
    return a;
  }

  ONIKA_HOST_DEVICE_FUNC inline Mat3d& operator -= ( Mat3d& a, const Mat3d& b)
  {
    a = a - b;
    return a;
  }

  ONIKA_HOST_DEVICE_FUNC inline Mat3d operator * ( const Mat3d& a, double b)
  {
    return Mat3d{
      a.m11*b,
      a.m12*b,
      a.m13*b,

      a.m21*b,
      a.m22*b,
      a.m23*b,

      a.m31*b,
      a.m32*b,
      a.m33*b };
  }
  ONIKA_HOST_DEVICE_FUNC inline Mat3d operator * ( double b, const Mat3d& a) { return a*b; }

/*
  ONIKA_HOST_DEVICE_FUNC inline Vec3d operator * ( const Mat3d& a, const Vec3d& b)
  {
   return { a.m11*b[0] + a.m12*b[1] + a.m13*b[2],
            a.m21*b[0] + a.m22*b[1] + a.m23*b[2],
            a.m31*b[0] + a.m32*b[1] + a.m33*b[2] };
  }
*/

  ONIKA_HOST_DEVICE_FUNC inline Mat3d operator / ( const Mat3d& a, double b)
  {
    return Mat3d{
      a.m11/b,
      a.m12/b,
      a.m13/b,

      a.m21/b,
      a.m22/b,
      a.m23/b,

      a.m31/b,
      a.m32/b,
      a.m33/b };
  }

  /// @brief Matrix transposition
  /// @tparam Class inside the matrix
  /// @param [in] m Input matrix
  /// @return Transposed matrix
  ONIKA_HOST_DEVICE_FUNC inline Mat3d transpose(const Mat3d& m)
  {
    return Mat3d{ m.m11, m.m21, m.m31, m.m12, m.m22, m.m32, m.m13, m.m23, m.m33 };
  }

  ONIKA_HOST_DEVICE_FUNC inline bool is_diagonal(const Mat3d& mat)
  {
    return mat.m12==0.0 && mat.m13==0.0 && mat.m21==0.0 && mat.m23==0.0 && mat.m31==0.0 && mat.m32==0.0 ;
  }

  ONIKA_HOST_DEVICE_FUNC inline bool is_triangular(const Mat3d& mat)
  {
    return mat.m12==0.0 && mat.m13==0.0 && mat.m23==0.0 ;
  }

  ONIKA_HOST_DEVICE_FUNC inline bool is_symmetric(const Mat3d& mat)
  {
    return mat.m12==mat.m21 && mat.m13==mat.m31 && mat.m23==mat.m32;
  }

  ONIKA_HOST_DEVICE_FUNC inline bool is_uniform_scale(const Mat3d& mat)
  {
    return is_diagonal(mat) && mat.m11==mat.m22 && mat.m11==mat.m33;
  }

  ONIKA_HOST_DEVICE_FUNC inline bool is_identity(const Mat3d& mat)
  {
    return mat.m11==1.0 && mat.m12==0.0 && mat.m13==0.0
        && mat.m21==0.0 && mat.m22==1.0 && mat.m23==0.0
        && mat.m31==0.0 && mat.m32==0.0 && mat.m33==1.0;
  }

  /// @brief determinant
  ONIKA_HOST_DEVICE_FUNC inline double determinant (const Mat3d& mat)
  {
    return (
      mat.m11 * mat.m22 * mat.m33 +
	    mat.m12 * mat.m23 * mat.m31 +
	    mat.m13 * mat.m21 * mat.m32 -
	    mat.m13 * mat.m22 * mat.m31 -
	    mat.m12 * mat.m21 * mat.m33 -
	    mat.m11 * mat.m23 * mat.m32 );
  }

  /// @brief comatrix
  ONIKA_HOST_DEVICE_FUNC inline Mat3d comatrix (const Mat3d& mat)
  {
    return Mat3d {
       mat.m22 * mat.m33 - mat.m32 * mat.m23,
		   mat.m31 * mat.m23 - mat.m21 * mat.m33,
		   mat.m21 * mat.m32 - mat.m31 * mat.m22,
		   mat.m32 * mat.m13 - mat.m12 * mat.m33,
		   mat.m11 * mat.m33 - mat.m31 * mat.m13,
		   mat.m31 * mat.m12 - mat.m11 * mat.m32,
		   mat.m12 * mat.m23 - mat.m22 * mat.m13,
		   mat.m21 * mat.m13 - mat.m11 * mat.m23,
		   mat.m11 * mat.m22 - mat.m21 * mat.m12
		   };
  }

  ONIKA_HOST_DEVICE_FUNC inline Mat3d multiply(const Mat3d &m, const Mat3d &n) {
    return Mat3d {
      m.m11*n.m11 + m.m12*n.m21 + m.m13*n.m31,
	    m.m11*n.m12 + m.m12*n.m22 + m.m13*n.m32,
	    m.m11*n.m13 + m.m12*n.m23 + m.m13*n.m33,
	    m.m21*n.m11 + m.m22*n.m21 + m.m23*n.m31,
	    m.m21*n.m12 + m.m22*n.m22 + m.m23*n.m32,
	    m.m21*n.m13 + m.m22*n.m23 + m.m23*n.m33,
	    m.m31*n.m11 + m.m32*n.m21 + m.m33*n.m31,
	    m.m31*n.m12 + m.m32*n.m22 + m.m33*n.m32,
	    m.m31*n.m13 + m.m32*n.m23 + m.m33*n.m33 };    
  }
  
  ONIKA_HOST_DEVICE_FUNC inline Mat3d operator * ( const Mat3d& a, const Mat3d& b)
  {
    return multiply( a, b );
  }

  ONIKA_HOST_DEVICE_FUNC inline bool operator == ( const Mat3d& a, const Mat3d& b)
  {
    return 
      a.m11==b.m11 &&
      a.m12==b.m12 &&
      a.m13==b.m13 &&

      a.m21==b.m21 &&
      a.m22==b.m22 &&
      a.m23==b.m23 &&

      a.m31==b.m31 &&
      a.m32==b.m32 &&
      a.m33==b.m33 ;
  }

  // component wise multiply
  ONIKA_HOST_DEVICE_FUNC inline Mat3d comp_multiply ( const Mat3d& a, const Mat3d& b)
  {
    return Mat3d{
      a.m11*b.m11,
      a.m12*b.m12,
      a.m13*b.m13,

      a.m21*b.m21,
      a.m22*b.m22,
      a.m23*b.m23,

      a.m31*b.m31,
      a.m32*b.m32,
      a.m33*b.m33 };
  }
  
  /// @brief inverse
  ONIKA_HOST_DEVICE_FUNC inline Mat3d inverse (const Mat3d& mat)
  {
    return transpose( comatrix(mat) ) / determinant(mat);
  }

  /*ONIKA_HOST_DEVICE_FUNC inline Vec3d inverse (const Vec3d& v)
  {
    return Vec3d{ 1.0/v.x, 1.0/v.y, 1.0/v.z};
  }*/
  
  ONIKA_HOST_DEVICE_FUNC inline Mat3d diag_matrix(const Vec3d& v)
  {
    return Mat3d{ v.x,0,0, 0,v.y,0, 0,0,v.z };
  }

  ONIKA_HOST_DEVICE_FUNC inline double trace_matrix(const Mat3d& mat)
  {
    return (mat.m11 + mat.m22 + mat.m33);
  }
  
  // use in tensorial product : Cij = AikBkj means Cij = sum over k (Aik * Bkj)
  ONIKA_HOST_DEVICE_FUNC inline Mat3d AikBkj (const Mat3d& A, const Mat3d& B) {
    return Mat3d {
      A.m11*B.m11 + A.m12*B.m21 + A.m13*B.m31,
      A.m11*B.m12 + A.m12*B.m22 + A.m13*B.m32,
      A.m11*B.m13 + A.m12*B.m23 + A.m13*B.m33,
      A.m21*B.m11 + A.m22*B.m21 + A.m23*B.m31,
      A.m21*B.m12 + A.m22*B.m22 + A.m23*B.m32,
      A.m21*B.m13 + A.m22*B.m23 + A.m23*B.m33,
      A.m31*B.m11 + A.m32*B.m21 + A.m33*B.m31,
      A.m31*B.m12 + A.m32*B.m22 + A.m33*B.m32,
      A.m31*B.m13 + A.m32*B.m23 + A.m33*B.m33};
  }

  ONIKA_HOST_DEVICE_FUNC inline bool is_zero(const Mat3d& mat)
  {
    return mat.m11==0 && mat.m12==0 && mat.m13==0
        && mat.m21==0 && mat.m22==0 && mat.m23==0
        && mat.m31==0 && mat.m32==0 && mat.m33==0 ;
  }

  inline double diff_l2_norm (const Mat3d& A, const Mat3d& B)
  {
    Mat3d D = A-B;
    Mat3d C = comp_multiply( D , D );
    return std::sqrt( C.m11+C.m12+C.m13 + C.m21+C.m22+C.m23 + C.m31+C.m32+C.m33 );
  }

  static inline double l2_norm (const Mat3d& mat)
  {
    return std::sqrt( mat.m11*mat.m11+mat.m12*mat.m12+mat.m13*mat.m13 + mat.m21*mat.m21+mat.m22*mat.m22+mat.m23*mat.m23 + mat.m31*mat.m31+mat.m32*mat.m32+mat.m33*mat.m33 );
  }

  static inline void save_nan(Mat3d& mat)
  {
    if(std::isnan(mat.m11) || std::isnan(mat.m13) || std::isnan(mat.m13) || std::isnan(mat.m21) || std::isnan(mat.m22) || std::isnan(mat.m23) || std::isnan(mat.m31) || std::isnan(mat.m32) || std::isnan(mat.m33)) mat = make_identity_matrix();
  }

  // conversion to/from std arrays
  inline std::array<ssize_t,3> to_array( const IJK& v ) { return {v.i,v.j,v.k}; }
  inline std::array<double,3> to_array( const Vec3d& v ) { return {v.x,v.y,v.z}; }

// this allows to specify Mat3d or Vec3d variable in an omp reduction clause
# pragma omp declare reduction(+:Mat3d:omp_out+=omp_in) initializer(omp_priv=Mat3d{0.,0.,0. ,0.,0.,0. ,0.,0.,0.})	
# pragma omp declare reduction(+:Vec3d:omp_out+=omp_in) initializer(omp_priv=Vec3d{0.,0.,0.})	

  // Fake Mat3d operations, to avoid computing anything
  template<class T> ONIKA_HOST_DEVICE_FUNC inline FakeMat3d& operator += (FakeMat3d& a, const T&) { return a; }
  ONIKA_HOST_DEVICE_FUNC inline constexpr bool is_zero(FakeMat3d) { return true; }
# pragma omp declare reduction(+:FakeMat3d:omp_out+=omp_in) initializer(omp_priv=FakeMat3d{})	
}

#include <exanb/core/bit_rotl.h> // optional replacement for C++20's std::rotl

namespace std
{
  template<> struct hash< exanb::IJK >
  {
    size_t operator () ( const exanb::IJK& v ) const
    {
      //std::hash<ssize_t> H{};
      return std::hash<ssize_t>{} ( v.i ^ exanb::bit_rotl(v.j,16) ^ exanb::bit_rotl(v.k,32) );
      // return H(v.i) ^ H(v.j) ^ H(v.k);
    }
  };

}

