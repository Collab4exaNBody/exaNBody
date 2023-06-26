#include <iostream>
#include <exanb/mpi/xs_data_move.h>

#include <cstdint>
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
    
    std::cout<<"MPI started, rank="<<rank<<", np="<<np<<std::endl;  std::cout.flush();
    
    // initialize Ids range
    XsDataMove::index_type IdMin = atoi( argv[1] );
    XsDataMove::index_type IdMax = atoi( argv[2] );
    XsDataMove::index_type IdRotation = atoi( argv[3] );
    assert( IdMax > IdMin );
    assert( IdRotation >= 0 );

    if( (IdMax-IdMin)/np < 4 ) { IdMax = IdMin + np*4; }
    XsDataMove::size_type IdRange = IdMax - IdMin;
    std::cout<<"P"<<rank<<": global Ids : ["<<IdMin<<";"<<IdMax<<"[ => "<<IdRange<<", rotation="<<IdRotation<<std::endl;  std::cout.flush();
    
    assert( (IdRange / np) >= 4 );
    
    XsDataMove::index_type beforeIdsStart = (IdRange*rank)/np;
    XsDataMove::index_type beforeIdsEnd = (IdRange*(rank+1))/np;
    XsDataMove::size_type beforeIdsCount = beforeIdsEnd - beforeIdsStart;

    XsDataMove::index_type afterIdsStart = beforeIdsStart;
    XsDataMove::index_type afterIdsEnd = beforeIdsEnd;
    if( rank%2==1)
    {
        if(afterIdsStart>0) --afterIdsStart;
        if(afterIdsEnd<IdRange) ++afterIdsEnd;
    }
    else
    {
        if(afterIdsStart>0 && afterIdsStart<IdRange) ++afterIdsStart;
        if(afterIdsEnd<IdRange && afterIdsEnd>0) --afterIdsEnd;
    }
    
    if( afterIdsStart < 0 ) afterIdsStart = 0;
    if( afterIdsEnd > IdRange ) afterIdsEnd = IdRange;
    XsDataMove::size_type afterIdsCount = afterIdsEnd - afterIdsStart;
    
    //std::cout<<"P"<<rank<<": before Ids : ["<<beforeIdsStart<<";"<<beforeIdsEnd<<"[ => "<<beforeIdsCount<<std::endl;  std::cout.flush();
    //std::cout<<"P"<<rank<<": after Ids : ["<<afterIdsStart<<";"<<afterIdsEnd<<"[ => "<<afterIdsCount<<std::endl;  std::cout.flush();
    
    XsDataMove::index_type* beforeIds = new XsDataMove::index_type [ beforeIdsCount ];
    XsDataMove::index_type* afterIds = new XsDataMove::index_type [ afterIdsCount ];
    XsDataMove::index_type* afterIdsCheck = new XsDataMove::index_type [ afterIdsCount ];
    
    for(XsDataMove::index_type i=0;i<beforeIdsCount;i++)
    {
      beforeIds[i] = IdMin + beforeIdsStart + i;
      assert( beforeIds[i]>=IdMin && beforeIds[i]<IdMax );
    }
    for(XsDataMove::index_type i=0;i<afterIdsCount;i++)
    {
      afterIdsCheck[i] = -1;
      afterIds[i] = IdMin + ( ( afterIdsStart + i + IdRotation ) % IdRange ) ;
      assert( afterIds[i]>=IdMin && afterIds[i]<IdMax );
    }

#   ifdef VERBOSE_DEBUG
    std::cout<<"P"<<rank<<":        Bedfore Ids:";
    for(int i=0;i<beforeIdsCount;i++)
    {
      std::cout<<" "<<beforeIds[i];
    }
    std::cout<<std::endl; std::cout.flush();
    std::cout<<"P"<<rank<<": Original after Ids:";
    for(int i=0;i<afterIdsCount;i++)
    {
      std::cout<<" "<<afterIds[i];
    }
    std::cout<<std::endl; std::cout.flush();
#   endif
    
    std::vector<int> send_indices;     // resized to localElementCountBefore, contain indices in 'before' array to pack into send buffer
    std::vector<int> send_count;       // resized to number of processors in comm, unit data element count (not byte size)
    std::vector<int> send_displ;       // resized to number of processors in comm, unit data element count (not byte size)
    std::vector<int> recv_indices;     // resized to localElementCountAfter
    std::vector<int> recv_count;       // resized to number of processors in comm, unit data element count (not byte size)
    std::vector<int> recv_displ;        // resized to number of processors in comm, unit data element count (not byte size)

    XsDataMove::communication_scheme_from_ids( MPI_COMM_WORLD, IdMin, IdMax, beforeIdsCount, beforeIds, afterIdsCount, afterIds,
                                               send_indices, send_count, send_displ, recv_indices, recv_count, recv_displ);
    
    assert( send_indices.size() == beforeIdsCount );
    assert( send_count.size() == np );
    assert( send_displ.size() == np );
    assert( recv_indices.size() == afterIdsCount );
    assert( recv_count.size() == np );
    assert( recv_displ.size() == np );
    
    // check that communication scheme produces valid data movement
    XsDataMove::data_move( MPI_COMM_WORLD, send_indices, send_count, send_displ, recv_indices, recv_count, recv_displ, beforeIds, afterIdsCheck );

#   ifdef VERBOSE_DEBUG
    std::cout<<"P"<<rank<<":    Check after Ids:";
    for(int i=0;i<afterIdsCount;i++)
    {
      std::cout<<" "<<afterIdsCheck[i];
    }
    std::cout<<std::endl; std::cout.flush();
#   endif

    for(int i=0;i<afterIdsCount;i++)
    {
      assert( afterIdsCheck[i] == afterIds[i] );
    }

    // test with arbitrary struct
    std::vector<MyStruct> myData(beforeIdsCount);
    for(int i=0;i<beforeIdsCount;i++)
    {
      myData[i] = MyStruct { beforeIds[i]*0.5 , beforeIds[i] , beforeIds[i]*2.0 };
    }
    myData.resize( std::max(beforeIdsCount,afterIdsCount) );
    XsDataMove::data_move( MPI_COMM_WORLD, send_indices, send_count, send_displ, recv_indices, recv_count, recv_displ, myData.data(), myData.data() );
    myData.resize(afterIdsCount);
    for(int i=0;i<afterIdsCount;i++)
    {
      assert( myData[i].id == afterIds[i] );
      assert( myData[i].id*0.5 == myData[i].x );
      assert( myData[i].id*2.0 == myData[i].y );
    }

    
    MPI_Finalize();

	return 0;
}
