#include <string>
#include <tuple>
#include <iostream>
#include <random>
#include <cmath>

#define _SoatlAliasTest 1

#include <onika/soatl/field_id.h>
#include <onika/soatl/field_arrays.h>
#include <onika/variadic_template_utils.h>
#include <onika/force_assert.h>

#include "declare_fields.h"

std::default_random_engine rng;

using namespace onika;

// WARNING: assumes that elements in arrays support 'operator = (const size_t&)' and 'operator == (const size_t&)'
template<typename FieldsT, typename... _ids>
static inline void check_field_arrays_aliasing( size_t N, FieldsT& field_arrays, const soatl::FieldId<_ids>& ... fids )
{
	std::uniform_int_distribution<> rndist(0,N*2);

#ifdef DEBUG
	std::cout<<"check_field_arrays_aliasing: align="<<field_arrays.alignment()<<", chunksize="<<field_arrays.chunksize()<<", using fields :"<<std::endl;
	TEMPLATE_LIST_BEGIN
		std::cout<<"\t"<< soatl::FieldId<_ids>::name() <<std::endl
	TEMPLATE_LIST_END
#endif

  for(size_t j=1;j<=N;j++)
  {
    field_arrays.resize(j);

		// test pointer alignment
		{
			const size_t alignment = field_arrays.alignment();
			void* ptr;
			size_t addr = 0;
			TEMPLATE_LIST_BEGIN
				ptr = field_arrays[fids] ,
				addr = reinterpret_cast<size_t>(ptr) ,
				ONIKA_FORCE_ASSERT( ( addr % alignment ) == 0 ) 
			TEMPLATE_LIST_END
		}

		// container 'c' must guarantee that acces beyond c.size() is valid up to the next chunk boundary
		// container only guarantees that read or write access beyond c.size() (up to the next chunk) but values in this area are undefined

#ifdef DEBUG
		std::cout<<"check_field_arrays_aliasing: a="<<field_arrays.alignment()
			 <<", c="<<field_arrays.chunksize()
			 <<", size="<<field_arrays.size()
			 <<", capacity="<<field_arrays.capacity()
			 <<", chunk boundary="<<field_arrays.chunk_ceil()<<std::endl;
#endif

		size_t k = 0;
		for(size_t i=0;i<j;i++)
    {
			TEMPLATE_LIST_BEGIN
				ONIKA_FORCE_ASSERT( field_arrays[fids] != nullptr ) ,
				field_arrays[fids][i] = static_cast<typename soatl::FieldId<_ids>::value_type>( k ) ,
				++k
			TEMPLATE_LIST_END
		}
		size_t chunk_ceil = field_arrays.chunk_ceil();
		// read from and write to the area beyond size() and up to next chunk boundary
		for(size_t i=j;i<chunk_ceil;i++)
                {
			TEMPLATE_LIST_BEGIN
				k += static_cast<size_t>( field_arrays[fids][i] ) ,
				field_arrays[fids][i] = static_cast<typename soatl::FieldId<_ids>::value_type>( k ) 
			TEMPLATE_LIST_END
		}

		// read back values in [0;size()[ to check values are still correct
		k=0;
		for(size_t i=0;i<j;i++)
                {
			bool value_ok = false;
			size_t findex = 0;
			TEMPLATE_LIST_BEGIN
				value_ok = field_arrays[fids][i] == static_cast<typename soatl::FieldId<_ids>::value_type>( k ) ,
				//std::cout<<"value["<<findex<<"]="<< (size_t)(field_arrays[fids][i] )<<std::endl ,
				ONIKA_FORCE_ASSERT(value_ok) ,
				++ findex ,
				++k
			TEMPLATE_LIST_END
		}
		
		// test that a resize keeps values
		size_t ns = rndist(rng);
		size_t cs = std::min( ns , j );
#ifdef DEBUG
		std::cout<<"resize from "<<j<<" to "<<ns<<", cs="<<cs<< std::endl;
#endif
		field_arrays.resize( ns );
		//field_arrays.force_reallocate();
		k=0;
		for(size_t i=0;i<cs;i++)
    {
			bool value_ok = false;
			size_t findex = 0;
			TEMPLATE_LIST_BEGIN
				value_ok = field_arrays[fids][i] == static_cast<typename soatl::FieldId<_ids>::value_type>( k ) ,
#ifdef DEBUG
				std::cout<<"value["<<_ids<<"]["<<i<<"] ="<< (size_t)( field_arrays[fids][i] )<<std::endl ,
#endif
				ONIKA_FORCE_ASSERT(value_ok) ,
				++ findex ,
				++k
			TEMPLATE_LIST_END
		}

	}

	// test that all pointers are 0 (nullptr) when capacity is 0
	{
		field_arrays.resize(0); // assumes that capacity is adjusted to 0 when container is resized to 0
		void* ptr;
		size_t findex = 0;
		TEMPLATE_LIST_BEGIN
			ptr = field_arrays[fids] ,
#ifdef DEBUG
			std::cout<<"ptr["<<findex<<"]="<<ptr<<std::endl ,
#endif
			++findex , 
			ONIKA_FORCE_ASSERT(ptr==nullptr) 
		TEMPLATE_LIST_END
	}
}

template<size_t A, size_t C>
static inline void test_hybrid_field_arrays_aliasing(size_t N)
{
	std::cout<<"test_hybrid_field_arrays_aliasing<"<<A<<","<<C<<">"<<std::endl;

	auto rx = particle_rx;
	auto ry = particle_ry;
	auto rz = particle_rz;
	//auto e = particle_e;
	auto atype = particle_atype;
	auto mid = particle_mid;
	auto tmp1 = particle_tmp1;
	auto tmp2 = particle_tmp2;
	auto dist = particle_dist;

	{
		auto field_arrays = soatl::make_hybrid_field_arrays( soatl::cst::align<A>(), soatl::cst::chunk<C>(), atype,rx,mid,ry,dist,rz,tmp1 );
		ONIKA_FORCE_ASSERT( field_arrays.alignment()==A && field_arrays.chunksize()==C );
		check_field_arrays_aliasing(N,field_arrays, atype,rx,mid,ry,dist,rz,tmp1 );
		check_field_arrays_aliasing(N,field_arrays, rx,ry,rz,atype,mid,dist,tmp1 );
		check_field_arrays_aliasing(N,field_arrays, atype,mid,dist,tmp1,rx,ry,rz );
	}

	{
		auto field_arrays = soatl::make_hybrid_field_arrays( soatl::cst::align<A>(), soatl::cst::chunk<C>(), atype,mid,dist,tmp2,tmp1,rx,ry,rz );
		ONIKA_FORCE_ASSERT( field_arrays.alignment()==A && field_arrays.chunksize()==C );
		check_field_arrays_aliasing(N,field_arrays, atype,rx,mid,ry,dist,rz,tmp1 );
		check_field_arrays_aliasing(N,field_arrays, rx,ry,rz,atype,mid,tmp2,dist,tmp1 );
		check_field_arrays_aliasing(N,field_arrays, atype,mid,dist,tmp1,rx,ry,tmp2,rz );
	}

	{
		auto field_arrays = soatl::make_hybrid_field_arrays( soatl::cst::align<A>(), soatl::cst::chunk<C>(), rx,ry,rz,atype,mid,dist,tmp2,tmp1 );
		ONIKA_FORCE_ASSERT( field_arrays.alignment()==A && field_arrays.chunksize()==C );
		check_field_arrays_aliasing(N,field_arrays, atype,tmp2,rx,mid,ry,dist,rz,tmp1 );
		check_field_arrays_aliasing(N,field_arrays, rx,ry,tmp2,rz,atype,mid,dist,tmp1 );
		check_field_arrays_aliasing(N,field_arrays, atype,mid,dist,tmp1,rx,ry,tmp2,rz );
	}

}


template<size_t A, size_t C>
static inline void test_packed_field_pointer_tuple()
{
	std::cout<<"test_packed_field_pointer_tuple<"<<A<<","<<C<<">"<<std::endl;

	auto rx = particle_rx;
	auto ry = particle_ry;
	auto rz = particle_rz;
	//auto e = particle_e;
	auto atype = particle_atype;
	auto mid = particle_mid;
	auto tmp1 = particle_tmp1;
	//auto tmp2 = particle_tmp2;
	auto dist = particle_dist;
	
	
	// check that a pointer tuple always initializes pointers to nullptr
	auto test_ptrs = soatl::make_field_pointer_tuple( soatl::cst::align<A>(), soatl::cst::chunk<C>(), atype,rx,ry,dist,rz,tmp1);
	ONIKA_FORCE_ASSERT( test_ptrs[atype] == nullptr );
	ONIKA_FORCE_ASSERT( test_ptrs[rx] == nullptr );
	ONIKA_FORCE_ASSERT( test_ptrs[ry] == nullptr );
	ONIKA_FORCE_ASSERT( test_ptrs[rz] == nullptr );
	ONIKA_FORCE_ASSERT( test_ptrs[dist] == nullptr );
	ONIKA_FORCE_ASSERT( test_ptrs[tmp1] == nullptr );


	auto field_arrays = soatl::make_hybrid_field_arrays( soatl::cst::align<A>(), soatl::cst::chunk<C>(), atype,rx,ry,dist,rz,tmp1 );
	field_arrays.resize( 100 );
	
	{
    //auto ptr_tuple1 = soatl::make_field_pointer_tuple( soatl::cst::align<A>(), soatl::cst::chunk<C>(), tmp1,atype,dist,rz );
    auto ptr_tuple2 = soatl::make_field_pointer_tuple( soatl::cst::align<A>(), soatl::cst::chunk<C>(), tmp1,atype,dist,rz );
    //field_arrays.capture_pointers_old( ptr_tuple1 );
    field_arrays.capture_pointers( ptr_tuple2 );
    //ONIKA_FORCE_ASSERT( ptr_tuple1 == ptr_tuple2 );
  }

	{ // mid doest not exist in source array, should remain null in pointer tuple
    //auto ptr_tuple1 = soatl::make_field_pointer_tuple( soatl::cst::align<A>(), soatl::cst::chunk<C>(), rz, mid, dist );
    auto ptr_tuple2 = soatl::make_field_pointer_tuple( soatl::cst::align<A>(), soatl::cst::chunk<C>(), rz, mid, dist );
    //field_arrays.capture_pointers_old( ptr_tuple1 );
    //ONIKA_FORCE_ASSERT( ptr_tuple1[mid] == nullptr );
    field_arrays.capture_pointers( ptr_tuple2 );
    ONIKA_FORCE_ASSERT( ptr_tuple2[mid] == nullptr );
    //ONIKA_FORCE_ASSERT( ptr_tuple1 == ptr_tuple2 );
  }


	{
    //auto ptr_tuple1 = soatl::make_field_pointer_tuple( soatl::cst::align<A>(), soatl::cst::chunk<C>(), dist,rz,rx );
    auto ptr_tuple2 = soatl::make_field_pointer_tuple( soatl::cst::align<A>(), soatl::cst::chunk<C>(), dist,rz,rx );
    //field_arrays.capture_pointers_old( ptr_tuple1 );
    field_arrays.capture_pointers( ptr_tuple2 );
    //ONIKA_FORCE_ASSERT( ptr_tuple1 == ptr_tuple2 );
  }
  
}

int main(int argc, char* argv[])
{
	int seed = 0;
	size_t N = 1063;

	if(argc>=2) { N=atoi(argv[1]); }
	if(argc>=3) { seed=atoi(argv[2]); }

  std::cout<<"SOATL Test : N="<<N<<" , seed="<<seed<<"\n";

	rng.seed( seed );

  test_packed_field_pointer_tuple<1,1>();
  test_packed_field_pointer_tuple<64,16>();

	test_hybrid_field_arrays_aliasing<1,1>(N);
	test_hybrid_field_arrays_aliasing<1,2>(N);
	test_hybrid_field_arrays_aliasing<1,3>(N);
	test_hybrid_field_arrays_aliasing<1,6>(N);
	test_hybrid_field_arrays_aliasing<1,16>(N);

	test_hybrid_field_arrays_aliasing<8,1>(N);
	test_hybrid_field_arrays_aliasing<8,2>(N);
	test_hybrid_field_arrays_aliasing<8,3>(N);
	test_hybrid_field_arrays_aliasing<8,6>(N);
	test_hybrid_field_arrays_aliasing<8,16>(N);

	test_hybrid_field_arrays_aliasing<64,1>(N);
	test_hybrid_field_arrays_aliasing<64,2>(N);
	test_hybrid_field_arrays_aliasing<64,3>(N);
	test_hybrid_field_arrays_aliasing<64,6>(N);
	test_hybrid_field_arrays_aliasing<64,16>(N);

	return 0;
}


