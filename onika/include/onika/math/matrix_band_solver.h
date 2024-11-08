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

#include <cstdio>
#include <cassert>
#include <vector>
#include <algorithm>

namespace onika { namespace math
{

  class MatrixBandSolver
  {
    std::vector< std::vector<double> > m_upper_band;
    std::vector< std::vector<double> > m_lower_band;
    int m_width = 0;
      
  public:
    MatrixBandSolver() = default;                 
    MatrixBandSolver(int dim, int n_u, int n_l);     
    ~MatrixBandSolver() = default;                    
    void resize(int dim, int n_u, int n_l);      
    inline int width() const { return m_width; }
    inline int upper_size() const { return m_upper_band.size()-1; }
    inline int lower_size() const { return m_lower_band.size()-1; }

    double at(int i, int j) const;
    double& at(int i, int j);
    
    double& diagonal_backup(int i);
    double  diagonal_backup(int i) const;
    void lu();
    std::vector<double> solve_r(const std::vector<double>& b) const;
    std::vector<double> solve_l(const std::vector<double>& b) const;
    std::vector<double> solve(const std::vector<double>& b);
  };

} } // end of onika::math namespace

