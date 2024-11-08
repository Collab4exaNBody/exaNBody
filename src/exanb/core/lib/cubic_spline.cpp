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

#include <exanb/core/spline.h>
#include <onika/test/unit_test.h>
#include <iostream>
#include <cmath>

namespace exanb
{

  void Spline::set_points(const std::vector<double>& in_x, const std::vector<double>& in_y)
  {
      assert(in_x.size()==in_y.size());
      assert(in_x.size()>2);
      const int n = in_x.size();
      std::vector< std::pair<double,double> > xy;
      for(int i=0; i<n; i++)
      {
        xy.push_back( { in_x[i] , in_y[i] } );
      }
      set_points( xy );
  }

  void Spline::set_points( std::vector< std::pair<double,double> > & xy)
  {
      const int n = xy.size();
      assert( n > 2 );
      
      std::sort( xy.begin() , xy.end() );
      m_x.clear();
      m_y.clear();
      for(const auto& p : xy)
      {
        m_x.push_back(p.first);
        m_y.push_back(p.second);
      }
      
      // TODO: maybe sort x and y, rather than returning an error
      for(int i=0; i<n-1; i++)
      {
          assert(m_x[i]<m_x[i+1]);
      }

      // setting up the matrix and right hand side of the equation system
      // for the parameters b[]
      onika::math::MatrixBandSolver A( n , 1 , 1 );
      std::vector<double> rhs( n , 0.0 );
      for(int i=1; i<n-1; i++)
      {
          A.at(i,i-1)=1.0/3.0*(m_x[i]-m_x[i-1]);
          A.at(i,i)=2.0/3.0*(m_x[i+1]-m_x[i-1]);
          A.at(i,i+1)=1.0/3.0*(m_x[i+1]-m_x[i]);
          rhs[i]=(m_y[i+1]-m_y[i])/(m_x[i+1]-m_x[i]) - (m_y[i]-m_y[i-1])/(m_x[i]-m_x[i-1]);
      }
      
      // boundary conditions
      // 2*b[0] = f''
      A.at(0,0)=2.0;
      A.at(0,1)=0.0;
      rhs[0]=0.0;
      
      // 2*b[n-1] = f''
      A.at(n-1,n-1)=2.0;
      A.at(n-1,n-2)=0.0;
      rhs[n-1]=0.0;
      
      // solve the equation system to obtain the parameters b[]
      m_b=A.solve(rhs);

      // calculate parameters a[] and c[] based on b[]
      m_a.assign( n , 0.0 );
      m_c.assign( n , 0.0 );
      for(int i=0; i<n-1; i++)
      {
          m_a[i]=1.0/3.0*(m_b[i+1]-m_b[i])/(m_x[i+1]-m_x[i]);
          m_c[i]=(m_y[i+1]-m_y[i])/(m_x[i+1]-m_x[i])
                 - 1.0/3.0*(2.0*m_b[i]+m_b[i+1])*(m_x[i+1]-m_x[i]);
      }
      
      // for left extrapolation coefficients
      m_b0 = m_b[0];
      m_c0 = m_c[0];

      // for the right extrapolation coefficients
      // f_{n-1}(x) = b*(x-x_{n-1})^2 + c*(x-x_{n-1}) + y_{n-1}
      const double h = m_x[n-1] - m_x[n-2];
      // m_b[n-1] is determined by the boundary condition
      m_a[n-1] = 0.0;
      m_c[n-1] = 3.0*m_a[n-2]*h*h+2.0*m_b[n-2]*h+m_c[n-2];   // = f'_{n-2}(x_{n-1})
  }

  double Spline::eval(double x) const
  {
      const size_t n = m_x.size();
      // find the closest point m_x[idx] < x, idx=0 even if x<m_x[0]
      const auto it = std::lower_bound( m_x.begin() , m_x.end() , x );
      const int idx = std::max( int(it-m_x.begin())-1 , 0 );
      const double h = x - m_x[idx];
      if( x < m_x[0] )
      {
          return (m_b0*h + m_c0)*h + m_y[0];
      }
      else if(x>m_x[n-1])
      {
          return (m_b[n-1]*h + m_c[n-1])*h + m_y[n-1];
      }
      else
      {
          return ((m_a[idx]*h + m_b[idx])*h + m_c[idx])*h + m_y[idx];
      }
  }
}


// ================== unit tests ============================

ONIKA_UNIT_TEST(cubic_spline_test)
{
  using namespace exanb;
  
  static const std::vector<double> R = { 0x1.203b0104c1f4fp+2, -0x1.840a4e04cbde4p+1, -0x1.5a45cbe2b2adp+5, -0x1.4e2579728c4e1p+6, -0x1.ef280cf3bf45ap+6, -0x1.4815503a791eap+7, -0x1.989699fb129a6p+7, -0x1.e917e3bbac163p+7, -0x1.1ccc96be22c9p+8, -0x1.450d3b9e6f86ep+8, -0x1.6d4de07ebc44cp+8, -0x1.958e855f0902ap+8, -0x1.bdcf2a3f55c09p+8, -0x1.e60fcf1fa27e7p+8, -0x1.072839fff79e3p+9, -0x1.1b488c701dfd2p+9, -0x1.2f68dee0445c1p+9, -0x1.438931506abbp+9, -0x1.57a983c09119fp+9, -0x1.6bc9d630b778ep+9, -0x1.7fea28a0ddd7dp+9, -0x1.940a7b110436cp+9, -0x1.a82acd812a95cp+9, -0x1.bc4b1ff150f4bp+9, -0x1.d06b72617753ap+9, -0x1.e48bc4d19db29p+9, -0x1.f8ac1741c4118p+9, -0x1.066634d8f5384p+10, -0x1.10765e110867bp+10, -0x1.1a8687491b973p+10, -0x1.2496b0812ec6ap+10, -0x1.2ea6d9b941f62p+10, -0x1.38b702f155259p+10, -0x1.42c72c2968551p+10, -0x1.4cd755617b849p+10, -0x1.56e77e998eb4p+10, -0x1.60f7a7d1a1e38p+10, -0x1.6b07d109b512fp+10, -0x1.7517fa41c8427p+10, -0x1.7f282379db71ep+10, -0x1.89384cb1eea16p+10, -0x1.934875ea01d0dp+10, -0x1.9d589f2215005p+10, -0x1.a768c85a282fdp+10, -0x1.b178f1923b5f4p+10, -0x1.bb891aca4e8ecp+10, -0x1.c599440261be3p+10, -0x1.cfa96d3a74edbp+10, -0x1.d9b99672881d2p+10, -0x1.e3c9bfaa9b4cap+10, -0x1.edd9e8e2ae7c2p+10, -0x1.f7ea121ac1ab9p+10, -0x1.00fd1da96a6d8p+11, -0x1.0605324574054p+11, -0x1.0b0d46e17d9dp+11, -0x1.10155b7d8734cp+11, -0x1.151d701990cc7p+11, -0x1.1a2584b59a643p+11, -0x1.1f2d9951a3fbfp+11, -0x1.2435adedad93bp+11, -0x1.293dc289b72b7p+11, -0x1.2e45d725c0c32p+11, -0x1.334debc1ca5aep+11, -0x1.3856005dd3f2ap+11, -0x1.3d5e14f9dd8a6p+11, -0x1.42662995e7221p+11, -0x1.476e3e31f0b9dp+11, -0x1.4c7652cdfa519p+11, -0x1.517e676a03e95p+11, -0x1.56867c060d811p+11, -0x1.5b8e90a21718cp+11, -0x1.6096a53e20b08p+11, -0x1.659eb9da2a484p+11, -0x1.6aa6ce7633ep+11, -0x1.6faee3123d77cp+11, -0x1.74b6f7ae470f7p+11, -0x1.79bf0c4a50a73p+11, -0x1.7ec720e65a3efp+11, -0x1.83cf358263d6bp+11, -0x1.88d74a1e6d6e6p+11, -0x1.8ddf5eba77062p+11, -0x1.92e77356809dep+11, -0x1.97ef87f28a35ap+11, -0x1.9cf79c8e93cd6p+11, -0x1.a1ffb12a9d651p+11, -0x1.a707c5c6a6fcdp+11, -0x1.ac0fda62b0949p+11, -0x1.b117eefeba2c5p+11, -0x1.b620039ac3c4p+11, -0x1.bb281836cd5bcp+11, -0x1.c0302cd2d6f38p+11, -0x1.c538416ee08b4p+11, -0x1.ca40560aea23p+11, -0x1.cf486aa6f3babp+11, -0x1.d4507f42fd527p+11, -0x1.d95893df06ea3p+11, -0x1.de60a87b1081fp+11, -0x1.e368bd171a19ap+11, -0x1.e870d1b323b16p+11, -0x1.ed78e64f2d492p+11 };
  static const double maxerr = 1.e-9;

  std::vector< std::pair<double,double> > P = { {5.5,2} , {0.23,1} , {1.2,4} , {0.66,0} , {-1.2,5}, {0.0,3} };
  Spline s;
  s.set_points( P );

/*
  std::cout << "static const std::vector<double> R = { ";
  for(int i=0;i<100;i++)
  {
    const double Xi = i*8.0 - 1.5;
    std::cout << std::hexfloat << s.eval(Xi) << ( (i!=99) ? ", " : " };\n" ) ;
  }
*/

  for(int i=0;i<100;i++)
  {
    const double Xi = i*8.0 - 1.5;
    const double Yi = s.eval( Xi );
    ONIKA_TEST_ASSERT( std::fabs(Yi-R[i]) < maxerr );
  }  

}

