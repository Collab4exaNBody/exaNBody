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
// this is a test, it always needs enabled asserts
#ifndef NDEBUG
#define NDEBUG 1
#endif

#include <onika/math/basic_types.h>
#include <exanb/core/geometry.h>
#include <onika/debug.h>

#include <iostream>
#include <random>
#include <cstdlib>

int main(int argc,char*argv[])
{
  
  using std::cout;
  using std::endl;

  size_t Nconfigs = 10000;
  size_t Nsamples = 10000;
  unsigned int seed = 0;
  if(argc>=2)
  {
    seed = atoi(argv[1]);
  }

  std::default_random_engine rng;
  rng.seed( seed );
  std::uniform_real_distribution<> rdist(-10.0,10.0);
  std::uniform_real_distribution<> rdist2(0.,1.);

  // test simple operators
  IJK loc = {3,-2,7};
  IJK loc2 = loc+1;
  cout<<loc.i<<' '<<loc.j<<' '<<loc.k<<" ; "<<loc2.i<<' '<<loc2.j<<' '<<loc2.k<<endl;
  ONIKA_FORCE_ASSERT( (loc.i+1)==loc2.i && (loc.j+1)==loc2.j && (loc.k+1)==loc2.k);
  
  // test min_distance and max_distance algorithms
  size_t n_intersect = 0;
  size_t n_order_x = 0;
  size_t n_order_y = 0;
  size_t n_order_z = 0;
  for(size_t i=0;i<Nconfigs;i++)
  {
    if( i%100 == 0 )
    {
      cout<<"[min|max]_distance_between : random configurations ... "<< (i*100)/Nconfigs << "%   \r"; cout.flush();
    }
    AABB A { {rdist(rng),rdist(rng),rdist(rng)} , {rdist(rng),rdist(rng),rdist(rng)} };
    AABB B { {rdist(rng),rdist(rng),rdist(rng)} , {rdist(rng),rdist(rng),rdist(rng)} };
    Vec3d p = { rdist(rng) , rdist(rng) , rdist(rng) };
    reorder_min_max(A);
    reorder_min_max(B);
    if( ! is_empty( intersection(A,B) ) )
    {
      ++ n_intersect;
      if( A.bmin.x < B.bmin.x ) { ++ n_order_x; }
      if( A.bmin.y < B.bmin.y ) { ++ n_order_y; }
      if( A.bmin.z < B.bmin.z ) { ++ n_order_z; }
    }
    Vec3d As = { A.bmax.x - A.bmin.x , A.bmax.y - A.bmin.y , A.bmax.z - A.bmin.z };
    Vec3d Bs = { B.bmax.x - B.bmin.x , B.bmax.y - B.bmin.y , B.bmax.z - B.bmin.z };

    double mindist = min_distance_between( A, B );
    double maxdist = max_distance_between( A, B );
    
    double mindist_pA = min_distance_between(p,A);
    double mindist_pB = min_distance_between(p,B);
    
    for(size_t j=0;j<Nsamples;j++)
    {
      Vec3d a = { rdist2(rng)*As.x+A.bmin.x , rdist2(rng)*As.y+A.bmin.y , rdist2(rng)*As.z+A.bmin.z };
      Vec3d b = { rdist2(rng)*Bs.x+B.bmin.x , rdist2(rng)*Bs.y+B.bmin.y , rdist2(rng)*Bs.z+B.bmin.z };
      ONIKA_FORCE_ASSERT( is_inside(A,a) );
      ONIKA_FORCE_ASSERT( is_inside(B,b) );
      double d = distance(a,b);
      ONIKA_FORCE_ASSERT( d >= mindist );
      ONIKA_FORCE_ASSERT( d <= maxdist );
      //d=d; // disable warning

      Vec3d a2 = { rdist2(rng)*As.x+A.bmin.x , rdist2(rng)*As.y+A.bmin.y , rdist2(rng)*As.z+A.bmin.z };
      double da = distance(a,a2);
      ONIKA_FORCE_ASSERT( da <= max_distance_inside(A) );
      //da=da; // disable warning
      
      ONIKA_FORCE_ASSERT( distance(p,a) >= mindist_pA );
      ONIKA_FORCE_ASSERT( distance(p,b) >= mindist_pB );
    }
  }
  cout<<"[min|max]_distance_between : random configurations ... 100%"<<endl;
  cout<<"intersecting box pairs (x/y/z ordered) = "<<n_intersect<<"/"<<n_order_x<<"/"<<n_order_y<<"/"<<n_order_z<<" over "<<Nconfigs<<endl;

  // test max_distance_inside
  for(size_t i=0;i<Nconfigs;i++)
  {
    if( i%100 == 0 )
    {
      cout<<"max_distance_inside : random configurations ... "<< (i*100)/Nconfigs << "%   \r"; cout.flush();
    }
    AABB A { {rdist(rng),rdist(rng),rdist(rng)} , {rdist(rng),rdist(rng),rdist(rng)} };
    reorder_min_max(A);
    Vec3d As = { A.bmax.x - A.bmin.x , A.bmax.y - A.bmin.y , A.bmax.z - A.bmin.z };
    for(size_t j=0;j<Nsamples;j++)
    {
      Vec3d a = { rdist2(rng)*As.x+A.bmin.x , rdist2(rng)*As.y+A.bmin.y , rdist2(rng)*As.z+A.bmin.z };
      Vec3d a2 = { rdist2(rng)*As.x+A.bmin.x , rdist2(rng)*As.y+A.bmin.y , rdist2(rng)*As.z+A.bmin.z };
      double da = distance(a,a2);
      ONIKA_FORCE_ASSERT( da <= max_distance_inside(A) );
      //da=da;
    }
  }
  cout<<"max_distance_inside : random configurations ... 100%"<<endl;

  return 0;
}


