#include <iostream>
#include <exanb/mpi/xs_data_move.h>

#include <cstdint>
#include <random>
#include <algorithm>
#include <mpi.h>
#include <stdlib.h> 
#include <assert.h>

struct MyStruct
{
  double x;
  uint64_t id;
  double y;
};


int main(int argc, char* argv[])
{
    MPI_Init(&argc,&argv);
    
    int rank = -1;
    int np = -1;
    MPI_Comm_rank(MPI_COMM_WORLD ,&rank);
    MPI_Comm_size(MPI_COMM_WORLD ,&np);
        
    // initialize Ids range
    int seed = atoi( argv[1] );    
    int N = atoi( argv[2] );
    int perms = atoi( argv[3] );
    std::cout<<"MPI started, rank="<<rank<<", np="<<np<<", seed="<<seed<<", N="<<N<<", permutations="<<perms<<std::endl;  std::cout.flush();
    
    std::mt19937 rng(seed);
    
    XsDataMove::index_type all_IdMin = seed;
    XsDataMove::index_type all_IdMax = all_IdMin+N;
    std::vector<XsDataMove::index_type> all_ids_before( N );
    for(int i=0;i<N;i++)
    {
      all_ids_before[i] = all_IdMin+i;
    }

    {
      std::uniform_int_distribution<> dist(0,N-1);
      for(int i=0;i<perms;i++)
      {
        int a = dist(rng);
        int b = dist(rng);
        std::swap( all_ids_before[a] , all_ids_before[b] );
      }
    }

    std::vector<int> procRanges(np+1);
    procRanges[0] = 0;
    procRanges[np] = N; // ensures we span the entire range
    {
      std::uniform_int_distribution<> dist(0,N);
      for(int i=0;i<(np-1);i++)
      {
        procRanges[i+1] = dist(rng);
      }
    }
    std::sort( procRanges.begin(), procRanges.end() );
    
    XsDataMove::index_type before_range_start = procRanges[rank];
    XsDataMove::index_type before_range_end = procRanges[rank+1];
    XsDataMove::size_type before_ids_count = before_range_end - before_range_start;
    XsDataMove::index_type* before_ids = all_ids_before.data() + before_range_start;
    
    // all_ids_after is equal to all_ids_before with perms permutations
    std::vector<XsDataMove::index_type> all_ids_after = all_ids_before;
    {
      std::uniform_int_distribution<> dist(0,N-1);
      for(int i=0;i<perms;i++)
      {
        int a = dist(rng);
        int b = dist(rng);
        std::swap( all_ids_after[a] , all_ids_after[b] );
      }
    }
    procRanges[0] = 0;
    procRanges[np] = N; // ensures we span the entire range
    {
      std::uniform_int_distribution<> dist(0,N);
      for(int i=0;i<(np-1);i++)
      {
        procRanges[i+1] = dist(rng);
      }
    }
    std::sort( procRanges.begin(), procRanges.end() );
    
    XsDataMove::index_type after_range_start = procRanges[rank];
    XsDataMove::index_type after_range_end = procRanges[rank+1];
    XsDataMove::size_type after_ids_count = after_range_end - after_range_start;
    XsDataMove::index_type* after_ids = all_ids_after.data() + after_range_start;

#   ifdef VERBOSE_DEBUG
    std::cout<<"P"<<rank<<":        Bedfore Ids:";
    for(int i=0;i<before_ids_count;i++)
    {
      std::cout<<" "<<before_ids[i];
    }
    std::cout<<std::endl; std::cout.flush();
    std::cout<<"P"<<rank<<": Original after Ids:";
    for(int i=0;i<after_ids_count;i++)
    {
      std::cout<<" "<<after_ids[i];
    }
    std::cout<<std::endl; std::cout.flush();
#   endif
    
    std::vector<int> send_indices;     // resized to localElementCountBefore, contain indices in 'before' array to pack into send buffer
    std::vector<int> send_count;       // resized to number of processors in comm, unit data element count (not byte size)
    std::vector<int> send_displ;       // resized to number of processors in comm, unit data element count (not byte size)
    std::vector<int> recv_indices;     // resized to localElementCountAfter
    std::vector<int> recv_count;       // resized to number of processors in comm, unit data element count (not byte size)
    std::vector<int> recv_displ;        // resized to number of processors in comm, unit data element count (not byte size)

    XsDataMove::communication_scheme_from_ids( MPI_COMM_WORLD, all_IdMin, all_IdMax, before_ids_count, before_ids, after_ids_count, after_ids,
                                               send_indices, send_count, send_displ, recv_indices, recv_count, recv_displ);
    
    assert( send_indices.size() == before_ids_count );
    assert( send_count.size() == np );
    assert( send_displ.size() == np );
    assert( recv_indices.size() == after_ids_count );
    assert( recv_count.size() == np );
    assert( recv_displ.size() == np );
    
    std::vector<XsDataMove::index_type> after_ids_check( after_ids, after_ids+after_ids_count );
    
    // check that communication scheme produces valid data movement
    XsDataMove::data_move( MPI_COMM_WORLD, send_indices, send_count, send_displ, recv_indices, recv_count, recv_displ, before_ids, after_ids_check.data() );

#   ifdef VERBOSE_DEBUG
    std::cout<<"P"<<rank<<":    Check after Ids:";
    for(int i=0;i<after_ids_count;i++)
    {
      std::cout<<" "<<after_ids_check[i];
    }
    std::cout<<std::endl; std::cout.flush();
#   endif

    for(int i=0;i<after_ids_count;i++)
    {
      assert( after_ids_check[i] == after_ids[i] );
    }


    // test with arbitrary struct
    std::vector<MyStruct> myData(before_ids_count);
    for(int i=0;i<before_ids_count;i++)
    {
      myData[i] = MyStruct { before_ids[i]*0.5 , before_ids[i] , before_ids[i]*2.0 };
    }
    myData.resize( std::max(before_ids_count,after_ids_count) );
    XsDataMove::data_move( MPI_COMM_WORLD, send_indices, send_count, send_displ, recv_indices, recv_count, recv_displ, myData.data(), myData.data() );
    myData.resize(after_ids_count);
    for(int i=0;i<after_ids_count;i++)
    {
      assert( myData[i].id == after_ids[i] );
      assert( myData[i].id*0.5 == myData[i].x );
      assert( myData[i].id*2.0 == myData[i].y );
    }


    MPI_Finalize();

	return 0;
}

