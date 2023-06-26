#pragma once

#include <exanb/core/basic_types_def.h>
#include <exanb/core/quaternion.h>
#include <cmath>
#include <onika/cuda/cuda.h>

namespace exanb
{
  ONIKA_HOST_DEVICE_FUNC inline Quaternion operator - (Quaternion& q)
  {
    return Quaternion{ -q.w, -q.x, -q.y, -q.z };
  }

  ONIKA_HOST_DEVICE_FUNC inline Quaternion vabs(const Quaternion& q)
  {
    return Quaternion{ fabs(q.w), fabs(q.x), fabs(q.y), fabs(q.z) };
  }

  ONIKA_HOST_DEVICE_FUNC inline Quaternion operator - (const Quaternion& q)
  {
    return Quaternion{ -q.w, -q.x, -q.y, -q.z };
  }

  ONIKA_HOST_DEVICE_FUNC inline Quaternion operator - (const Quaternion& q, const Quaternion& r)
  {
    return Quaternion{ q.w-r.w, q.x-r.x, q.y-r.y, q.z-r.z };
  }

  ONIKA_HOST_DEVICE_FUNC inline Quaternion operator + (const Quaternion& q, const Quaternion& r)
  {
    return Quaternion{ q.w+r.w, q.x+r.x, q.y+r.y, q.z+r.z };
  }

  ONIKA_HOST_DEVICE_FUNC inline Quaternion operator * (const Quaternion& q, const Quaternion& r)
  {
    return Quaternion{ q.w*r.w, q.x*r.x, q.y*r.y, q.z*r.z };
  }

  ONIKA_HOST_DEVICE_FUNC inline Quaternion operator / (const Quaternion& q, const Quaternion& r)
  {
    return Quaternion{ q.w/r.w, q.x/r.x, q.y/r.y, q.z/r.z };
  }

  ONIKA_HOST_DEVICE_FUNC inline Quaternion operator * (const Quaternion& q, double scale)
  {
    return Quaternion{ q.w*scale, q.x*scale, q.y*scale, q.z*scale };
  }
  ONIKA_HOST_DEVICE_FUNC inline Quaternion operator / (const Quaternion& q, double scale)
  {
    return Quaternion{ q.w/scale, q.x/scale, q.y/scale, q.z/scale };
  }
  ONIKA_HOST_DEVICE_FUNC inline Quaternion operator * (double scale, const Quaternion& q)
  {
    return Quaternion{ q.w*scale, q.x*scale, q.y*scale, q.z*scale };
  }
  ONIKA_HOST_DEVICE_FUNC inline Quaternion operator / (double scale, const Quaternion& q)
  {
    return Quaternion{ q.w/scale, q.x/scale, q.y/scale, q.z/scale };
  }

  ONIKA_HOST_DEVICE_FUNC inline Quaternion& operator /= (Quaternion& q, double b)
  {
    q.w /= b;
    q.x /= b;
    q.y /= b;
    q.z /= b;
    return q;
  }
  
  ONIKA_HOST_DEVICE_FUNC inline Quaternion& operator += (Quaternion& q, Quaternion& r)
  {
    q.w += r.w;
    q.x += r.x;
    q.y += r.y;
    q.z += r.z;
    return q;
  }
  
  ONIKA_HOST_DEVICE_FUNC inline double norm(const Quaternion& q)
  {
    return ( sqrt( q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z ) );
  }
  
  ONIKA_HOST_DEVICE_FUNC inline Quaternion normalize(const Quaternion& q)
  {
    return ( q / norm( q ) );
  }

  ONIKA_HOST_DEVICE_FUNC inline Quaternion get_conjugated(const Quaternion& q)
  {
    Quaternion ret;
    ret.w = q.w;
    ret.x = -q.x;
    ret.y = -q.y;
    ret.z = -q.z;
    return ret;
  }
  
  /// @brief Return the time derivative of the quaternion \f$ \dot{q} = \frac{1}{2} \omega q\f$
  ONIKA_HOST_DEVICE_FUNC inline Quaternion dot(const Quaternion& q, const Vec3d& omega)
  {
    Quaternion ret;
    ret.x = 0.5 * (omega.y * q.z - omega.z * q.y + omega.x * q.w);
    ret.y = 0.5 * (omega.z * q.x - omega.x * q.z + omega.y * q.w);
    ret.z = 0.5 * (omega.x * q.y - omega.y * q.x + omega.z * q.w);
    ret.w = -0.5 * (exanb::dot(omega,Vec3d{q.x,q.y,q.z})); // real dot product
    return ret;
  }
  
  /// @brief Return d2Q/dt2
  ONIKA_HOST_DEVICE_FUNC inline Quaternion ddot (const Quaternion& q, const Vec3d& vrot, const Vec3d& arot)
  {
/*
    	  Quaternion ret;
    double omega2_2 = 0.5 * exanb::dot(vrot,vrot);
    auto xyz = Vec3d{q.x,q.y,q.z};
    auto tmp = 0.5 * (q.w * arot + exanb::cross(arot, xyz) -
                        omega2_2 * xyz);
    ret.x = tmp.x;
    ret.y = tmp.y;
    ret.z = tmp.z;
    ret.w = -0.5 * (exanb::dot(arot , xyz) + omega2_2) ;

    return ret;
*/

    auto xyz = Vec3d{q.x,q.y,q.z};
    auto tmp1 = q.w * arot + cross(arot, xyz);
    auto tmp2 = - dot(arot , xyz);
    //quat q1(s * domega + cross(domega, v), -domega * v); // remark: ctor is quat(v, s)
    Quaternion q1 = {tmp1.x, tmp1.y, tmp1.z, tmp2}; 
    Vec3d c = cross(vrot, xyz);
    //quat q2(-(omega * v) * omega + cross(omega, s * omega + c), omega * (s * omega + c));
    tmp1 = - dot(vrot, xyz) * vrot + cross (vrot, q.w*vrot + c);
    tmp2 = dot(vrot , (q.w * vrot + c)); 
    Quaternion q2 = {tmp1.x, tmp1.y, tmp1.z, tmp2};
    q1 += (q2 = q2 * 0.5);
    q1 = q1 * 0.5;
    return q1;
  }

  // Rotation of a vector by a quat
  ONIKA_HOST_DEVICE_FUNC inline Vec3d operator * (const Quaternion& q, const Vec3d& v)
  {
    const auto xyz = Vec3d{q.x,q.y,q.z};
    auto qv = cross(xyz, v);
    auto qqv = cross(xyz, qv);
    return v + qv*2.0*q.w + qqv*2.0;
  }
}

