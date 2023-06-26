#include <string>
#include <iostream>
#include <random>
#include <cmath>
#include <chrono>
#include <atomic>
#include <omp.h>

#include <onika/soatl/field_id.h>
#include <onika/soatl/field_arrays.h>
#include <onika/variadic_template_utils.h>
#include <onika/soatl/compute.h>

#include "declare_fields.h"

#ifndef TEST_ALIGNMENT
#define TEST_ALIGNMENT 64
#endif

#ifndef TEST_CHUNK_SIZE
#define TEST_CHUNK_SIZE 16
#endif

#ifndef TEST_USE_SIMD
#define TEST_USE_SIMD 1
#endif

#ifndef TEST_DOUBLE_PRECISION
#define TEST_DOUBLE_PRECISION 1
#endif

#ifndef TEST_ENABLE_OPENMP
#define TEST_ENABLE_OPENMP 1
#endif

#if TEST_DOUBLE_PRECISION
auto field_e = particle_e;
auto field_rx = particle_rx;
auto field_ry = particle_ry;
auto field_rz = particle_rz;
#else
auto field_e = particle_e_f;
auto field_rx = particle_rx_f;
auto field_ry = particle_ry_f;
auto field_rz = particle_rz_f;
#endif

std::default_random_engine rng;

using namespace onika;

template<typename ArraysT, typename idDist, typename idRx, typename idRy, typename idRz>
inline double benchmark(ArraysT& arrays, size_t N, soatl::FieldId<idDist> dist, soatl::FieldId<idRx> rx, soatl::FieldId<idRy> ry, soatl::FieldId<idRz> rz)
{
  static constexpr size_t nCycles = 10;
  using DistT = typename soatl::FieldId<idDist>::value_type;
  using PosT = typename soatl::FieldId<idRx>::value_type;

  std::uniform_real_distribution<> rdist(0.0,1.0);
  double result = 0.0;

  std::chrono::nanoseconds timens(0);

  for(size_t cycle=0;cycle<nCycles;cycle++)
  {
	  arrays.resize(N);
	  for(size_t i=0;i<N;i++)
	  {
		  arrays[rx][i] = rdist(rng);
		  arrays[ry][i] = rdist(rng);
		  arrays[rz][i] = rdist(rng);
    }

    PosT ax = arrays[rx][0];
    PosT ay = arrays[ry][0];
    PosT az = arrays[rz][0];

#define COMPUTE_KERNEL \
  x = x - ax; \
	y = y - ay; \
	z = z - az; \
	d = std::sqrt( x*x + y*y + z*z ); \
	x /= d; \
	y /= d; \
	z /= d; \
	d += x/(y*z)

    auto t1 = std::chrono::high_resolution_clock::now();
#   if TEST_USE_SIMD
#     if TEST_ENABLE_OPENMP
	    soatl::parallel_apply_simd( [ax,ay,az](DistT& d, PosT x, PosT y, PosT z) { COMPUTE_KERNEL; }
		                            , arrays, dist, rx, ry, rz );
#     else
	    soatl::apply_simd( [ax,ay,az](DistT& d, PosT x, PosT y, PosT z) { COMPUTE_KERNEL; }
		                   , arrays, dist, rx, ry, rz );
#     endif
#   else
#     if TEST_ENABLE_OPENMP
	    soatl::parallel_apply( [ax,ay,az](DistT& d, PosT x, PosT y, PosT z) { COMPUTE_KERNEL; }
		                       , arrays, dist, rx, ry, rz );
#     else
	    soatl::apply( [ax,ay,az](DistT& d, PosT x, PosT y, PosT z) { COMPUTE_KERNEL; }
		              , arrays, dist, rx, ry, rz );
#     endif
#   endif
    auto t2 = std::chrono::high_resolution_clock::now();
    timens += t2-t1;

    // try to optimize this out, compiler !
    for(size_t i=0;i<N;i++)
    {
      if(rdist(rng)>0.999) { result += arrays[dist][i]; }
    }
	}

  std::cout<<"time = "<<timens.count()/nCycles<<std::endl;

  return result;
}

int main(int argc, char* argv[])
{
	//static constexpr size_t S=300;
	int seed = 0;
	size_t N = 10000;

	if(argc>=2)
	{
	  N = atoi(argv[2]);
	}

	if(argc>=3)
	{
	  seed = atoi(argv[3]);
	}

  //if(arraysImpl == STATIC_PACKED_FIELD_ARRAYS) { N = S; }

  std::cout<<"SIMD arch="<<memory::simd_arch();
# if TEST_DOUBLE_PRECISION
    std::cout<<" a="<< memory::SimdRequirements<double>::alignment <<" c=" << memory::SimdRequirements<double>::chunksize;
# else
    std::cout<<" a="<< memory::SimdRequirements<float>::alignment <<" c=" << memory::SimdRequirements<float>::chunksize;
# endif

# if TEST_ENABLE_OPENMP
    std::atomic<int> n_threads(0);
#   pragma omp parallel
    {
      n_threads = omp_get_num_threads();
    }
    std::cout<<", omp="<<n_threads;
# endif

# if TEST_DOUBLE_PRECISION
  std::cout<<", double";
# else
  std::cout<<", float";
# endif

# if TEST_USE_SIMD 
  std::cout<<", vec";
# else
  std::cout<<", scal";
# endif

  std::cout<<", N="<<N<<", seed="<<seed<<std::endl;

	rng.seed( seed );

  double result = 0.0;
  auto arrays = soatl::make_hybrid_field_arrays( soatl::cst::align<TEST_ALIGNMENT>(), soatl::cst::chunk<TEST_CHUNK_SIZE>(), field_rx, field_ry, field_rz, field_e);
  result = benchmark(arrays,N,field_e,field_rx,field_ry,field_rz);
  std::cout<<"result = "<<result<<std::endl;

	return 0;
}


