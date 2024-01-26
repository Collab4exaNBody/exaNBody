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

