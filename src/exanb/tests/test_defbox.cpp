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
#include <exanb/defbox/deformation.h>
#include <exanb/core/math_utils.h>
#include <exanb/core/basic_types_operators.h>
#include <exanb/core/basic_types_stream.h>

#include <iostream>
#include <random>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <assert.h>



/*
static inline bool check_identity( const Mat3d& mat )
{
  static constexpr double epsilon = 1.e-12;
  static constexpr Mat3d id { 1.,0.,0., 0.,1.,0., 0.,0.,1. };
  
  double diff = std::sqrt( norm2( column1(mat)-column1(id) ) + norm2( column2(mat)-column2(id) ) + norm2( column3(mat)-column3(id) ) );
  if(diff >= epsilon) { std::cout << "identity failed: " << diff << std::endl; }
  return diff < epsilon;
}
*/

int main(int argc, char* argv[])
{
  using std::endl;
  using std::cout;
  
  size_t Nconfigs = 1000;
  size_t Nsamples = 10000;
  
  unsigned int seed = 0;
  if(argc>=2)
  {
    seed = atoi(argv[1]);
  }

  std::default_random_engine rng;
  rng.seed( seed );
  
  size_t failure1 = 0;
  size_t failure2 = 0;

/*
  {
    Mat3d mat { 1.,1.,0., 0.,1.,0., 0.,0.,1. };
    double min_scale=0.0;
    double max_scale=std::numeric_limits<double>::max();
    matrix_scale_min_max( mat, min_scale, max_scale ); 
    cout << "scale=["<<min_scale<<";"<<max_scale<<", det="<<determinant(mat) <<endl;
  }
  return 0;
*/
  for(size_t j=0;j<Nconfigs;j++)
  {
/*    if( j%100 == 0 )
    {
      cout<<"defbox rcut adjust ... "<< (j*100)/Nconfigs << "%   \r"; cout.flush();
    }
  */  
    
    std::uniform_real_distribution<double> rdist(-1.0,1.0);
    std::uniform_real_distribution<double> uniform01(0.0,1.0);

    //assert( defbox_check_angles( defbox.m_angles ) );

    //double rcut = ( defbox.m_extension.x + defbox.m_extension.y + defbox.m_extension.z ) / 8.0;

    Vec3d v1 { rdist(rng), rdist(rng), rdist(rng) };
    Vec3d v2 { rdist(rng), rdist(rng), rdist(rng) };
    Vec3d v3 { rdist(rng), rdist(rng), rdist(rng) };
    Mat3d mat { v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, v3.x, v3.y, v3.z };

#   ifndef NDEBUG
    Vec3d vx{1.,0.,0.};
    Vec3d vy{0.,1.,0.};
    Vec3d vz{0.,0.,1.};
#   endif

    assert( mat*vx == column1(mat) );
    assert( mat*vy == column2(mat) );
    assert( mat*vz == column3(mat) );

    // eigen value test
    //Vec3d Q[3];
    //double W[3];
    //auto M2 = transpose(mat) * mat;
    //symmetric_matrix_eigensystem(M2,Q,W);
   
    double min_scale=0.0;
    double max_scale=std::numeric_limits<double>::max();
    matrix_scale_min_max( mat, min_scale, max_scale);

    //auto [min_scale,max_scale] = matrix_min_max_scale_iter(mat);
    
    double approx_min_scale=std::numeric_limits<double>::max();
    double approx_max_scale=0.0;

    size_t fail1=0;  
    size_t fail2=0;
    //std::uniform_real_distribution<double> uniform01(0.0,1.0);
    for(size_t i=0;i<Nsamples;i++)
    {
      double theta = 2 * M_PI * uniform01(rng);
      double phi = acos(1 - 2 * uniform01(rng));
      double x = sin(phi) * cos(theta);
      double y = sin(phi) * sin(theta);
      double z = cos(phi);
      Vec3d r { x , y , z };
      assert( fabs(norm(r)-1.0) < 1.e-10 );
      
      Vec3d rx = mat * r;
      double scale = norm(rx);
      
      bool test1 = ( scale >= min_scale );
      bool test2 = ( scale <= max_scale );
      if( !test1 ) { ++fail1; }
      if( !test2 ) { ++fail2; }
      /*if( !test1 || !test2 )
      {
        cout<<"failed for r="<<r<<", rx="<<rx<<", scale="<<scale<<", range=["<<min_scale<<";"<<max_scale<<"], det="<<determinant(mat) << std::endl;
      }*/
      approx_min_scale = std::min( approx_min_scale , scale );
      approx_max_scale = std::max( approx_max_scale , scale );
    }

    if( (fail1+fail2) > 0 ) cout << "scale=["<<min_scale<<";"<<max_scale<<"], approx=["<<approx_min_scale<<";"<<approx_max_scale<<"], failures="<<fail1<<"/"<<fail2<<" over "<<Nsamples<<endl;
    failure1 += fail1;
    failure2 += fail2;
  }
  cout << "failure ratio = "<<(failure1+failure2) / static_cast<double>(Nsamples*Nconfigs) <<", f1="<<failure1<<", f2="<<failure2<< endl;

  assert( (failure1+failure2) == 0 );

  return 0;
}

