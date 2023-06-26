#pragma once

#include <exanb/defbox/deformation.h>
#include <exanb/core/basic_types_operators.h>

#include <cmath>
#include <cassert>

#include <iostream>
#include <exanb/core/basic_types_stream.h>

namespace exanb
{

  /// @brief Compute the volume of a parallelepiped
  static inline double defbox_volume(const Vec3d& len, double al, double bt, double gm)
  {
    return std::sqrt( 1. - std::cos(al) * std::cos(al) - std::cos(bt) * std::cos(bt) - std::cos(gm) * std::cos(gm) +
                      2. * std::cos(al) * std::cos(bt) * std::cos(gm)
                    ) * len.x * len.y * len.z;
  }
  
  static inline bool defbox_check_angles(Vec3d angles)
  {
    static constexpr double twoPi = 2. * M_PI;

    const double al = angles.x;
    const double bt = angles.y;
    const double gm = angles.z;

    double test = al + bt + gm;
    if (test < 0. || test > twoPi) return false;
    test = -al + bt + gm;
    if (test < 0. || test > twoPi) return false;
    test = al - bt + gm;
    if (test < 0. || test > twoPi) return false;
    test = al + bt - gm;
    if (test < 0. || test > twoPi) return false;
    
    return true;
  }

  /// @brief Get distances between planes in parallelepiped
  static inline Vec3d defbox_distances(const Mat3d& pm)
  {
    Vec3d v1 = column1(pm) ;
    Vec3d v2 = column2(pm) ;
    Vec3d v3 = column3(pm) ;
    
    Vec3d c1 = cross(v2, v3);
    Vec3d c2 = cross(v3, v1);
    Vec3d c3 = cross(v1, v2);

    double h1 = std::fabs(dot(c1, v1)) / norm(c1);
    double h2 = std::fabs(dot(c2, v2)) / norm(c2);
    double h3 = std::fabs(dot(c3, v3)) / norm(c3);

    return Vec3d{ h1, h2, h3 };
  }

  static inline Mat3d angles_to_matrix( const Vec3d& angles )
  {
    const double al = angles.x;
    const double bt = angles.y;
    const double gm = angles.z;
    
    // compute pm
    Mat3d pm;

    /* pm.m11 = 1.; */
    /* pm.m21 = 0.; */
    /* pm.m31 = 0.; */
   
    /* pm.m12 = std::cos(al); */
    /* pm.m22 = std::sin(al); */
    /* pm.m32 = 0.; */
   
    /* pm.m13 = std::sin(bt) * std::cos(gm); */
    /* pm.m23 = std::cos(bt) * std::sin(gm); */
    /* pm.m33 = std::sin(bt) * std::sin(gm); */
    /* double n = std::sqrt(pm.m13 * pm.m13 + pm.m23 * pm.m23 + pm.m33 * pm.m33); */
    /* pm.m13 /= n; */
    /* pm.m23 /= n; */
    /* pm.m33 /= n; */

    const double cosAlpha = std::cos(al);
    const double cosBeta = std::cos(bt);
    const double cosGamma = std::cos(gm);
    const double sinGamma = std::sin(gm);
    Vec3d a,b,c;
    
    a.x = 1.;
    a.y = 0.;
    a.z = 0.;
   
    b.x = cosGamma;
    b.y = sinGamma;
    b.z = 0.;

    c.y = (cosAlpha - cosGamma*cosBeta)/sinGamma;
    c.x = cosBeta;
    c.z = std::sqrt(1. - c.x*c.x - c.y*c.y);

    pm.m11 = a.x;
    pm.m12 = a.y;
    pm.m13 = a.z;
    pm.m21 = b.x;
    pm.m22 = b.y;
    pm.m23 = b.z;
    pm.m31 = c.x;
    pm.m32 = c.y;
    pm.m33 = c.z;
    
    return transpose(pm);
  }

  static inline Mat3d deformation_to_matrix( const Deformation& defbox )
  {
    return multiply( angles_to_matrix( defbox.m_angles ) , diag_matrix( defbox.m_extension ) );
  }

}
