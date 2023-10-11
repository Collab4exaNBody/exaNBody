#include <exanb/core/matrix_band_solver.h>

namespace exanb
{

  MatrixBandSolver::MatrixBandSolver(int dim, int n_u, int n_l)
  {
    resize(dim, n_u, n_l);
  }
  
  void MatrixBandSolver::resize(int dim, int n_u, int n_l)
  {
    assert(dim>0);
    assert(n_u>=0);
    assert(n_l>=0);
    m_upper_band.resize(n_u+1);
    m_lower_band.resize(n_l+1);
    for(size_t i=0; i<m_upper_band.size(); i++)
    {
      m_upper_band[i].resize(dim);
    }
    for(size_t i=0; i<m_lower_band.size(); i++)
    {
      m_lower_band[i].resize(dim);
    }
  }
  
  int MatrixBandSolver::dim() const
  {
    if( ! m_upper_band.empty() ) return m_upper_band[0].size();
    else return 0;
  }

  double & MatrixBandSolver::at(int i, int j)
  {
    const int k = j - i;
    assert( (i>=0) && (i<dim()) && (j>=0) && (j<dim()) );
    assert( (-lower_size()<=k) && (k<=upper_size()) );
    if( k >= 0 ) return m_upper_band[k][i];
    else	       return m_lower_band[-k][i];
  }

  double MatrixBandSolver::at(int i, int j) const
  {
    const int k = j - i;
    assert( (i>=0) && (i<dim()) && (j>=0) && (j<dim()) );
    assert( (-lower_size()<=k) && (k<=upper_size()) );
    if(k>=0)   return m_upper_band[k][i];
    else	    return m_lower_band[-k][i];
  }

  // backup diag in m_lower_band
  double MatrixBandSolver::diagonal_backup(int i) const
  {
    assert( (i>=0) && (i<dim()) );
    return m_lower_band[0][i];
  }
  double & MatrixBandSolver::diagonal_backup(int i)
  {
    assert( (i>=0) && (i<dim()) );
    return m_lower_band[0][i];
  }

  // LR-Decomposition of a band matrix
  void MatrixBandSolver::lu()
  {
    for(int i=0; i<dim(); i++)
    {
      assert(at(i,i)!=0.0);
      diagonal_backup(i)=1.0/at(i,i);
      const int j_min = std::max(0,i-lower_size());
      const int j_max = std::min(dim()-1,i+upper_size());
      for(int j=j_min; j<=j_max; j++)
      {
        at(i,j) *= diagonal_backup(i);
      }
      at(i,i)=1.0;
    }

    // Gauss LR Decomposition
    for(int k=0; k<dim(); k++)
    {
      const int i_max = std::min(dim()-1,k+lower_size());
      for(int i=k+1; i<=i_max; i++)
      {
        assert(at(k,k)!=0.0);
        const double x = -at(i,k) / at(k,k);
        at(i,k )= -x;
        const int j_max = std::min(dim()-1,k+upper_size());
        for(int j=k+1; j<=j_max; j++)
        {
          at(i,j) = at(i,j) + x*at(k,j);
        }
      }
    }
  }
  
  // left solve
  std::vector<double> MatrixBandSolver::solve_l(const std::vector<double>& b) const
  {
    assert( dim() == static_cast<ssize_t>(b.size()) );
    std::vector<double> x( dim() , 0.0 );
    for(int i=0; i<dim(); i++)
    {
      double sum = 0;
      const int j_start = std::max(0,i-lower_size());
      for(int j=j_start; j<i; j++) sum += at(i,j)*x[j];
      x[i]=(b[i]*diagonal_backup(i)) - sum;
    }
    return x;
  }
  
  // right solve
  std::vector<double> MatrixBandSolver::solve_r(const std::vector<double>& b) const
  {
    assert( dim() == static_cast<ssize_t>(b.size()) );
    std::vector<double> x( dim() , 0.0 );
    int j_stop;
    double sum;
    for(int i=dim()-1; i>=0; i--)
    {
      sum=0;
      j_stop=std::min(dim()-1,i+upper_size());
      for(int j=i+1; j<=j_stop; j++) sum += at(i,j)*x[j];
      x[i]=( b[i] - sum ) / at(i,i);
    }
    return x;
  }

  std::vector<double> MatrixBandSolver::solve(const std::vector<double>& b)
  {
    assert( dim() == static_cast<ssize_t>(b.size()) );
    lu();
    return solve_r( solve_l(b) );
  }

} // namespace exanb

