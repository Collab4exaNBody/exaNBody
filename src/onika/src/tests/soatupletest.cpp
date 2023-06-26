#include <string>
#include <iostream>
#include <random>
#include <cmath>

#include <onika/soatl/field_id.h>
#include <onika/soatl/field_arrays.h>
#include <onika/variadic_template_utils.h>

#include "declare_fields.h"

std::default_random_engine rng;

using namespace onika;

template<typename... ids>
inline std::ostream& operator << (std::ostream& out, const soatl::FieldTuple<ids...>& value)
{
  TEMPLATE_LIST_BEGIN
    out << (double) value[ soatl::FieldId<ids>() ]<<" "
  TEMPLATE_LIST_END
  return out;
}

int main(int argc, char* argv[])
{
  static_assert( soatl::find_index_of_id<__particle_atype,__particle_atype,__particle_rx,__particle_mid,__particle_ry,__particle_rz>::index == 0 , "find_index_of_id is wrong" );
  static_assert( soatl::find_index_of_id<__particle_rx,__particle_atype,__particle_rx,__particle_mid,__particle_ry,__particle_rz>::index == 1 , "find_index_of_id is wrong" );
  static_assert( soatl::find_index_of_id<__particle_mid,__particle_atype,__particle_rx,__particle_mid,__particle_ry,__particle_rz>::index == 2 , "find_index_of_id is wrong" );
  static_assert( soatl::find_index_of_id<__particle_ry,__particle_atype,__particle_rx,__particle_mid,__particle_ry,__particle_rz>::index == 3 , "find_index_of_id is wrong" );
  static_assert( soatl::find_index_of_id<__particle_rz,__particle_atype,__particle_rx,__particle_mid,__particle_ry,__particle_rz>::index == 4 , "find_index_of_id is wrong" );

  int seed = 0;
  size_t N = 100;

  if(argc>=2) { N=atoi(argv[1]); }
  if(argc>=3) { seed=atoi(argv[2]); }

  assert( N > 0 && "N must be strictly greater than 0");
  
  rng.seed( seed );

  auto rx = particle_rx;
  auto ry = particle_ry;
  auto rz = particle_rz;
  auto e = particle_e;
  auto atype = particle_atype;
  auto mid = particle_mid;
  //auto tmp1 = particle_tmp1;
  //auto tmp2 = particle_tmp2;
  auto dist = particle_dist;

  auto cell_arrays1 = soatl::make_hybrid_field_arrays( rx,ry,rz,e,dist );
  auto cell_arrays2 = soatl::make_hybrid_field_arrays( atype,rx,mid,ry,rz );

  assert( cell_arrays1.size() == 0 );
  assert( cell_arrays1.capacity() == 0 );
  assert( cell_arrays1[rx] == nullptr );
  assert( cell_arrays1[ry] == nullptr );
  assert( cell_arrays1[rz] == nullptr );
  assert( cell_arrays1[e] == nullptr );
  assert( cell_arrays1[dist] == nullptr );

  assert( cell_arrays2.size() == 0 );
  assert( cell_arrays2.capacity() == 0 );
  assert( cell_arrays2[rx] == nullptr );
  assert( cell_arrays2[ry] == nullptr );
  assert( cell_arrays2[rz] == nullptr );
  assert( cell_arrays2[atype] == nullptr );
  assert( cell_arrays2[mid] == nullptr );
  
  std::cout<<"initialization with fixed value"<<std::endl;  std::cout.flush();
  cell_arrays1.assign(N, cell_arrays1.make_tuple(0, 2.0, 3.0, 4.0, 5.0f) );
  cell_arrays2.assign(N, cell_arrays2.make_tuple(1, 2.0, 3, 4.0, 5.0) );
  for(size_t i=0;i<N;i++)
  {
    std::cout << "arrays1["<<i<<"] = " << cell_arrays1[i] << std::endl;
  }
  for(size_t i=0;i<N;i++)
  {
    std::cout << "arrays2["<<i<<"] = " << cell_arrays2[i] << std::endl;
  }
  cell_arrays1.clear();
  cell_arrays2.clear();

  // all pointers are null and size and capacity are 0 when resize(0) (or clear) is called
  assert( cell_arrays1.size() == 0 );
  assert( cell_arrays1.capacity() == 0 );
  assert( cell_arrays1[rx] == nullptr );
  assert( cell_arrays1[ry] == nullptr );
  assert( cell_arrays1[rz] == nullptr );
  assert( cell_arrays1[e] == nullptr );
  assert( cell_arrays1[dist] == nullptr );

  assert( cell_arrays2.size() == 0 );
  assert( cell_arrays2.capacity() == 0 );
  assert( cell_arrays2[rx] == nullptr );
  assert( cell_arrays2[ry] == nullptr );
  assert( cell_arrays2[rz] == nullptr );
  assert( cell_arrays2[atype] == nullptr );
  assert( cell_arrays2[mid] == nullptr );

  std::cout<<"initialization with random values"<<std::endl;  std::cout.flush();
  std::uniform_real_distribution<> rdist(0.0,1.0);
  for(size_t i=0;i<N;i++)
  {
    cell_arrays1.push_back( cell_arrays1.make_tuple( rdist(rng), rdist(rng), rdist(rng), rdist(rng), rdist(rng) ) );
    cell_arrays2.push_back( cell_arrays2.make_tuple( static_cast<unsigned int>( rdist(rng)*50 ), rdist(rng), static_cast<unsigned int>(rdist(rng)*500), rdist(rng), rdist(rng) ) );
  }

  for(size_t i=0;i<N;i++)
  {
    std::cout << "arrays1["<<i<<"] = " << cell_arrays1[i] << std::endl;
  }
  for(size_t i=0;i<N;i++)
  {
    std::cout << "arrays2["<<i<<"] = " << cell_arrays2[i] << std::endl;
  }
  
  auto mytuple = soatl::make_field_tuple( rx,rz,dist,atype );
  std::cout << mytuple << std::endl;

  mytuple.copy_or_zero_fields( cell_arrays1[N-1] );
  std::cout << mytuple << std::endl;

  mytuple.copy_or_zero_fields( cell_arrays2[N-1] );
  std::cout << mytuple << std::endl;
  
  mytuple.copy_existing_fields( cell_arrays1[0] );
  mytuple.copy_existing_fields( cell_arrays2[0] );
  std::cout << mytuple << std::endl;
  
  { auto tmp = cell_arrays2[N-1]; tmp.copy_or_zero_fields(cell_arrays1[N-1]); cell_arrays2.set_tuple(N-1,tmp); }
  std::cout << cell_arrays2[N-1] << std::endl;

  { auto tmp = cell_arrays1[0]; tmp.copy_or_zero_fields(cell_arrays2[0]); cell_arrays1.set_tuple(0,tmp); }
  cell_arrays1[0].copy_existing_fields(cell_arrays2[0]);
  std::cout << cell_arrays1[0] << std::endl;
  
  return 0;
}

