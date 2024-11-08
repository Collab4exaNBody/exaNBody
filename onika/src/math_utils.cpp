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

#include <onika/math/math_utils.h>
#include <onika/math/basic_types_operators.h>
#include <onika/log.h>
#include <onika/test/unit_test.h>

#include <cmath>
#include <cassert>
#include <cstdlib>
#include <algorithm>

namespace onika { namespace math
{

  int dsyevj3(double A[3][3], double Q[3][3], double w[3]);

  void symmetric_matrix_eigensystem(const Mat3d& A, Vec3d Q[3], double w[3])
  {
    // std::cout<<"A : det="<<determinant(A)<<", sym="<<std::boolalpha<<is_symmetric(A)<<", tri="<<std::boolalpha<<is_triangular(A)<<", diag="<<std::boolalpha<<is_diagonal(A)<<std::endl;    
    assert( is_symmetric(A) );

    double _A[3][3];
    double _Q[3][3] = { {0.,0.,0.} , {0.,0.,0.} , {0.,0.,0.} };

    w[0] = 0.0;
    w[1] = 0.0;
    w[2] = 0.0;
    
    _A[0][0] = A.m11;
    _A[0][1] = A.m12;
    _A[0][2] = A.m13;

    _A[1][0] = A.m21;
    _A[1][1] = A.m22;
    _A[1][2] = A.m23;

    _A[2][0] = A.m31;
    _A[2][1] = A.m32;
    _A[2][2] = A.m33;
    
    dsyevj3(_A,_Q,w);

    Q[0] = Vec3d { _Q[0][0] , _Q[1][0] , _Q[2][0] };
    Q[1] = Vec3d { _Q[0][1] , _Q[1][1] , _Q[2][1] };
    Q[2] = Vec3d { _Q[0][2] , _Q[1][2] , _Q[2][2] };

    // -------------------------
    if( fabs(w[0]) > fabs(w[1]) ) { std::swap(w[0],w[1]); std::swap(Q[0],Q[1]); }
    if( fabs(w[1]) > fabs(w[2]) ) { std::swap(w[1],w[2]); std::swap(Q[1],Q[2]); }
    if( fabs(w[0]) > fabs(w[1]) ) { std::swap(w[0],w[1]); std::swap(Q[0],Q[1]); }
  }

  /* inspired from
    https://en.wikipedia.org/wiki/Power_iteration */
  void matrix_scale_min_max(const Mat3d & mat, double& scale_min, double& scale_max)
  {
    constexpr int niter = 100;
    constexpr int R = 4;
    constexpr int nvec = R*R*6;
    Vec3d b[nvec];
    Vec3d bi[nvec];
    int k=0;
    for(int i=0;i<R;i++)
    for(int j=0;j<R;j++)
    {
      b[k++] = Vec3d{  1.0 , i/(double)R , j/(double)R };
      b[k++] = Vec3d{ -1.0 , i/(double)R , j/(double)R };
      b[k++] = Vec3d{ i/(double)R ,  1.0 , j/(double)R };
      b[k++] = Vec3d{ i/(double)R , -1.0 , j/(double)R };
      b[k++] = Vec3d{ i/(double)R , j/(double)R ,  1.0 };
      b[k++] = Vec3d{ i/(double)R , j/(double)R , -1.0 };
    }
    assert(k==nvec);
    for(k=0;k<nvec;k++)
    {
      b[k] = b[k] / norm(b[k]);
      bi[k] = b[k];
    }
    auto imat = inverse(mat);
    auto A2 = transpose(mat) * mat;
    auto iA2 = transpose(imat) * imat;
    double conv = 0.0;
    int iter = 0;
    do
    {
      conv = 0.0;
      for(k=0;k<nvec;k++)
      {
        auto oldbk = b[k];
        auto oldbik = bi[k];
        b[k] = A2 * b[k];
        bi[k] = iA2 * bi[k];
        b[k] = b[k] / norm(b[k]);
        bi[k] = bi[k] / norm(bi[k]);
        conv += norm( b[k] - oldbk );
        conv += norm( bi[k] - oldbik );
      }
      ++iter;
    }
    while( conv > 1e-10 && iter<niter );
    double maxscale = std::sqrt( dot(b[0],A2*b[0])/dot(b[0],b[0]) );
    double imaxscale = std::sqrt( dot(bi[0],iA2*bi[0])/dot(bi[0],bi[0]) );
    for(k=1;k<nvec;k++)
    {
      maxscale = std::max( maxscale , std::sqrt( dot(b[k],A2*b[k])/dot(b[k],b[k]) ) );
      imaxscale = std::max( imaxscale , std::sqrt( dot(bi[k],iA2*bi[k])/dot(bi[k],bi[k]) ) );
    }
    scale_min = 1.0 / imaxscale;
    scale_max = maxscale;
  }

/* sphere parametric surface */
  Vec3d unitvec_uv( double u, double v )
  {
    double theta = 2 * M_PI * u;
    double phi = std::acos(1 - 2 * v);
    double x = std::sin(phi) * std::cos(theta);
    double y = std::sin(phi) * std::sin(theta);
    double z = std::cos(phi);
    return Vec3d { x , y , z };
  }


// Macros
#define SQR(x)      ((x)*(x))                        // x^2 

// ----------------------------------------------------------------------------
int dsyevj3(double A[3][3], double Q[3][3], double w[3])
// ----------------------------------------------------------------------------
// Calculates the eigenvalues and normalized eigenvectors of a symmetric 3x3
// matrix A using the Jacobi algorithm.
// The upper triangular part of A is destroyed during the calculation,
// the diagonal elements are read but not destroyed, and the lower
// triangular elements are not referenced at all.
// ----------------------------------------------------------------------------
// Parameters:
//   A: The symmetric input matrix
//   Q: Storage buffer for eigenvectors
//   w: Storage buffer for eigenvalues
// ----------------------------------------------------------------------------
// Return value:
//   0: Success
//  -1: Error (no convergence)
// ----------------------------------------------------------------------------
{
  const int n = 3;
  double sd, so;                  // Sums of diagonal resp. off-diagonal elements
  double s, c, t;                 // sin(phi), cos(phi), tan(phi) and temporary storage
  double g, h, z, theta;          // More temporary storage
  double thresh;
  
  // Initialize Q to the identitity matrix
#ifndef EVALS_ONLY
  for (int i=0; i < n; i++)
  {
    Q[i][i] = 1.0;
    for (int j=0; j < i; j++)
      Q[i][j] = Q[j][i] = 0.0;
  }
#endif

  // Initialize w to diag(A)
  for (int i=0; i < n; i++)
    w[i] = A[i][i];

  // Calculate SQR(tr(A))  
  sd = 0.0;
  for (int i=0; i < n; i++)
    sd += fabs(w[i]);
  sd = SQR(sd);
 
  // Main iteration loop
  for (int nIter=0; nIter < 50; nIter++)
  {
    // Test for convergence 
    so = 0.0;
    for (int p=0; p < n; p++)
      for (int q=p+1; q < n; q++)
        so += fabs(A[p][q]);
    if (so == 0.0)
      return 0;

    if (nIter < 4)
      thresh = 0.2 * so / SQR(n);
    else
      thresh = 0.0;

    // Do sweep
    for (int p=0; p < n; p++)
      for (int q=p+1; q < n; q++)
      {
        g = 100.0 * fabs(A[p][q]);
        if (nIter > 4  &&  fabs(w[p]) + g == fabs(w[p])
                       &&  fabs(w[q]) + g == fabs(w[q]))
        {
          A[p][q] = 0.0;
        }
        else if (fabs(A[p][q]) > thresh)
        {
          // Calculate Jacobi transformation
          h = w[q] - w[p];
          if (fabs(h) + g == fabs(h))
          {
            t = A[p][q] / h;
          }
          else
          {
            theta = 0.5 * h / A[p][q];
            if (theta < 0.0)
              t = -1.0 / (sqrt(1.0 + SQR(theta)) - theta);
            else
              t = 1.0 / (sqrt(1.0 + SQR(theta)) + theta);
          }
          c = 1.0/sqrt(1.0 + SQR(t));
          s = t * c;
          z = t * A[p][q];

          // Apply Jacobi transformation
          A[p][q] = 0.0;
          w[p] -= z;
          w[q] += z;
          for (int r=0; r < p; r++)
          {
            t = A[r][p];
            A[r][p] = c*t - s*A[r][q];
            A[r][q] = s*t + c*A[r][q];
          }
          for (int r=p+1; r < q; r++)
          {
            t = A[p][r];
            A[p][r] = c*t - s*A[r][q];
            A[r][q] = s*t + c*A[r][q];
          }
          for (int r=q+1; r < n; r++)
          {
            t = A[p][r];
            A[p][r] = c*t - s*A[q][r];
            A[q][r] = s*t + c*A[q][r];
          }

          // Update eigenvectors
#ifndef EVALS_ONLY          
          for (int r=0; r < n; r++)
          {
            t = Q[r][p];
            Q[r][p] = c*t - s*Q[r][q];
            Q[r][q] = s*t + c*Q[r][q];
          }
#endif
        }
      }
  }

  return -1;
}

} } // end of onika::math namesapce


// ************** Unit tests ******************

ONIKA_UNIT_TEST(matrix_scale_min_max)
{
    
    {
      using namespace onika::math;
      Mat3d m = { 1,0,0, 0,1,0, 0,0,1 };
      double smin, smax;
      matrix_scale_min_max(m,smin,smax);
      ONIKA_TEST_ASSERT( smin==1.0 && smax==1.0 );
    }

    {    
      using namespace onika::math;
      Mat3d m = { 0.5,0,0, 0,1,0, 0,0,2.0 };
      double smin, smax;
      matrix_scale_min_max(m,smin,smax);
      ONIKA_TEST_ASSERT( smin==0.5 && smax==2.0 );
    }
}


