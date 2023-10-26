#pragma once

#include <cassert>
#include <vector>
#include <algorithm>
#include <exanb/core/matrix_band_solver.h>

namespace exanb
{

  // Cubic spline interpolation
  class Spline
  {
    std::vector<double> m_x,m_y; // Spline coordinates
    std::vector<double> m_a,m_b,m_c; // cubic form coefficients : a.X^3 + b.X^2 + C.X
    double  m_b0, m_c0; // extrapolation

    friend struct ReadOnlySpline; 
  public:
    // set default boundary condition to be zero curvature at both ends
    Spline() = default;

    void set_points(const std::vector<double>& x, const std::vector<double>& y);
    void set_points(std::vector< std::pair<double,double> > & xy);
    double eval(double x) const;
  };

} // namespace exanb

