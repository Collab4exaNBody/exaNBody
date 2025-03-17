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

#include <exanb/amr/amr_grid.h>
#include <iostream>

namespace exanb
{

  template<typename GridT>
  static inline void print_grid_stats( const GridT& grid)
  {
    using std::cout;
    using std::endl;

    ssize_t max_part = -1;
    ssize_t min_part = 1000000000000ll;
    size_t total_particles = 0;
    auto cells = grid.cells();
    size_t n_cells = grid.number_of_cells();  
    for(size_t i=0;i<n_cells;i++)
    {
      ssize_t n = cells[i].size();
      total_particles += n;
      max_part = std::max( max_part , n );
      min_part = std::min( min_part , n );
    }
    cout<<"particles per cell (min/max) = "<<min_part<<" / "<<max_part<<endl;
    cout<<"number of cells = "<< grid.number_of_cells() << endl;
    cout<<"number of particles = "<<total_particles<<endl;  
  }

  template<typename GridT>
  static inline void print_grid_amr_stats( const GridT& grid, const AmrGrid& amr)
  {  
    using std::cout;
    using std::endl;

    ssize_t max_part = -1;
    ssize_t min_part = 1000000000000ll;
    ssize_t max_sg = -1;
    ssize_t min_sg = exanb::MAX_SUB_GRID_RESOLUTION + 1;
    size_t total_particles = 0;
    auto cells = grid.cells();
    size_t n_cells = grid.number_of_cells();  
    double sg_avg_density = 0.;
    for(size_t i=0;i<n_cells;i++)
    {
      ssize_t n = cells[i].size();
      total_particles += n;
      max_part = std::max( max_part , n );
      min_part = std::min( min_part , n );
      ssize_t r = amr.cell_resolution(i);
      sg_avg_density += n / static_cast<double>(r*r*r);
      max_sg = std::max( max_sg , r );
      min_sg = std::min( min_sg , r );
    }
    sg_avg_density /= n_cells;
    cout<<"particles per cell (min/max) = "<<min_part<<" / "<<max_part<<endl;
    cout<<"cell resolution (min/max) = "<<min_sg<<" / "<<max_sg<<endl;
    cout<<"avg sub grid density = "<<sg_avg_density<<endl;
    cout<<"number of cells = "<< grid.number_of_cells() << endl;
    cout<<"number of particles = "<<total_particles<<endl;  
  }

  template<typename GridT, class = AssertGridHasFields< GridT, field::_ax, field::_ay, field::_az, field::_ep > >
  static inline void print_energy_sum(GridT& grid)
  {
    using std::cout;
    using std::endl;
    auto cells = grid.cells();
    size_t n_cells = grid.number_of_cells();
    double sum_e = 0.;
    double sum_ax = 0.;
    double sum_ay = 0.;
    double sum_az = 0.;
    for(size_t i=0;i<n_cells;i++)
    {
      size_t n = cells[i].size();
      const double* ep = cells[i][field::ep];
      double* ax = cells[i][field::ax];
      double* ay = cells[i][field::ay];
      double* az = cells[i][field::az];
      for(size_t j=0;j<n;j++)
      {
        sum_ax += ax[j];
        sum_ay += ay[j];
        sum_az += az[j];
        sum_e += ep[j];
      }
    }
    double norm_a = std::sqrt( sum_ax*sum_ax + sum_ay*sum_ay + sum_az*sum_az );
    cout.precision(20);
    cout<<"sum(e)="<<sum_e<<", norm(a)="<<norm_a<< endl;
    cout.precision(6);
  }

  template<typename GridT , class = AssertGridHasFields< GridT, field::_ax, field::_ay, field::_az, field::_ep > >
  static inline void zero_energy(GridT& grid)
  {
    using std::cout;
    using std::endl;
    auto cells = grid.cells();
    size_t n_cells = grid.number_of_cells();
    for(size_t i=0;i<n_cells;i++)
    {
      size_t n = cells[i].size();
      double* ep = cells[i][field::ep];
      double* ax = cells[i][field::ax];
      double* ay = cells[i][field::ay];
      double* az = cells[i][field::az];
      for(size_t j=0;j<n;j++)
      {
        ep[j] = 0.;
        ax[j] = 0.;
        ay[j] = 0.;
        az[j] = 0.;
      }
    }
  }

}
