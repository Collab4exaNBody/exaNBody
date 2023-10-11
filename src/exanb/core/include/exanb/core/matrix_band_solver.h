#pragma once

#include <cstdio>
#include <cassert>
#include <vector>
#include <algorithm>

namespace exanb
{

  class MatrixBandSolver
  {
    std::vector< std::vector<double> > m_upper_band; 
    std::vector< std::vector<double> > m_lower_band;  
  public:
    MatrixBandSolver() = default;                 
    MatrixBandSolver(int dim, int n_u, int n_l);     
    ~MatrixBandSolver() = default;                    
    void resize(int dim, int n_u, int n_l);      
    int dim() const;                              
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

} // namespace exanb

