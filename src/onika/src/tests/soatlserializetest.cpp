#include <string>
#include <tuple>
#include <iostream>
#include <random>
#include <cmath>
#include <typeinfo>
#include <fstream>

#include <onika/soatl/field_id.h>
#include <onika/soatl/packed_field_arrays.h>
#include <onika/soatl/field_arrays.h>
#include <onika/variadic_template_utils.h>

#include "declare_fields.h"

std::default_random_engine rng;

using namespace onika;

template<typename T, typename id>
static inline void print_field_info(const T& arrays, soatl::FieldId<id> f)
{
	auto ptr = arrays[f];
	std::cout<<soatl::FieldId<id>::name()<<" : array type is "<<typeid(ptr).name()<<std::endl;
}

int main(int argc, char* argv[])
{
	int seed = 0;
	size_t N = 10000;

	if(argc>=2) { seed=atoi(argv[1]); }

	rng.seed( seed );

	auto rx = particle_rx;
	auto ry = particle_ry;
	auto rz = particle_rz;
	auto e = particle_e;
	auto atype = particle_atype;
	auto mid = particle_mid;

	auto in_arrays = soatl::make_hybrid_field_arrays( soatl::cst::align<64>(), soatl::cst::chunk<8>(), e,atype,rx,mid,ry,rz );
	auto serialize_arrays = soatl::make_hybrid_field_arrays( soatl::cst::align<1>(), soatl::cst::chunk<1>(), e,atype,rx,mid,ry,rz );
	auto out_arrays = soatl::make_hybrid_field_arrays( soatl::cst::align<64>(), soatl::cst::chunk<8>(), e,atype,rx,mid,ry,rz );

	std::cout<<"resize arrays to "<<N<<std::endl;  std::cout.flush();
	in_arrays.resize(N);

	std::cout<<"initialize values"<<std::endl;  std::cout.flush();
	std::uniform_real_distribution<> rdist(0.0,1.0);
	for(size_t i=0;i<N;i++)
	{
		in_arrays[rx][i] = rdist(rng);
		in_arrays[ry][i] = rdist(rng);
		in_arrays[rz][i] = rdist(rng);
		in_arrays[e][i] = rdist(rng);
		in_arrays[atype][i] = static_cast<unsigned int>( rdist(rng)*50 );
		in_arrays[mid][i] = static_cast<unsigned int>( rdist(rng)*500 );
	}
	
	std::cout<<"serialize arrays"<<std::endl;
	serialize_arrays.resize(N);
	std::cout<<"memory usage = "<<serialize_arrays.memory_bytes()<<std::endl;
	std::cout<<"data size = "<<serialize_arrays.storage_size()<<std::endl;

	// all these variants need to be tested
	soatl::copy( serialize_arrays , in_arrays, 0, N/2 );
	soatl::copy( serialize_arrays , in_arrays, N/2, N/2, rx, ry );
	soatl::copy( serialize_arrays , in_arrays, 0, N, rx, ry );
	soatl::copy( serialize_arrays , in_arrays, rz,e,atype,mid );

	soatl::copy( serialize_arrays , in_arrays );

	{
		std::ofstream fout("serialize.dat");
		fout.write( (const char*) serialize_arrays.storage_ptr() , serialize_arrays.storage_size() );
	}
	serialize_arrays.resize(0);
	serialize_arrays.resize(N);
	{
		uint8_t* ptr = (uint8_t*) serialize_arrays.storage_ptr();
		for(size_t i=0;i<serialize_arrays.storage_size();i++) { ptr[i]=0; }
		std::ifstream fin("serialize.dat");
		fin.read( (char*) serialize_arrays.storage_ptr() , serialize_arrays.storage_size() );
	}
	
	out_arrays.resize(N);
	soatl::copy( out_arrays, serialize_arrays );
	serialize_arrays.resize(0);

	for(size_t i=0;i<N;i++)
	{
		assert( in_arrays[rx][i] == out_arrays[rx][i] );
		assert( in_arrays[ry][i] == out_arrays[ry][i] );
		assert( in_arrays[rz][i] == out_arrays[rz][i] );
		assert( in_arrays[e][i] == out_arrays[e][i] );
		assert( in_arrays[atype][i] == out_arrays[atype][i] );
		assert( in_arrays[mid][i] == out_arrays[mid][i] );
	}

	std::cout<<"serialization ok"<<std::endl;
	return 0;
}


