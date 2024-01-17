#include <iostream>
#include <cstdint>
#include <vector>

#include <mpi.h>
#include <assert.h>

#include <exanb/mpi/xs_data_move_types.h>
#include <exanb/mpi/xs_data_move_internal_types.h>
#include <exanb/mpi/xs_data_move_range_utils.h>
#include <exanb/mpi/xs_data_move.h>

namespace XsDataMove
{

/*
  allIdMin : min id for both IdsBefore and IdsAfter
  allIdMax : max id for both IdsBefore and IdsAfter
*/
  
void communication_scheme_from_ids(
	// SYMETRIC IN, common across all processes 
	MPI_Comm comm,
	id_type allIdMin, // included
	id_type allIdMax, // excluded

	// IN, differ across processes
	size_type localElementCountBefore,
	const id_type* localIdsBefore,
	size_type localElementCountAfter,
	const id_type* localIdsAfter,

	// OUT
	std::vector<int>& send_indices,     // resized to localElementCountBefore, contain indices in 'before' array to pack into send buffer
	std::vector<int>& send_count,       // resized to number of processors in comm, unit data element count (not byte size)
    std::vector<int>& send_displ,       // resized to number of processors in comm, unit data element count (not byte size)
    std::vector<int>& recv_indices,     // resized to localElementCountAfter
    std::vector<int>& recv_count,       // resized to number of processors in comm, unit data element count (not byte size)
    std::vector<int>& recv_displ        // resized to number of processors in comm, unit data element count (not byte size)
	)
{
	assert( localIdsBefore != nullptr || localElementCountBefore==0 );
	assert( localIdsAfter != nullptr || localElementCountAfter==0 );
	assert( localElementCountBefore >= 0 );
	assert( localElementCountAfter >= 0 );

	int rank = -1;
	int nProcs = -1;
	MPI_Comm_rank(comm,&rank);
	MPI_Comm_size(comm,&nProcs);
	assert( rank>=0 && rank<nProcs );
	assert( nProcs>=1 );
    
    // ******* per range Id classification ********

    // count how many ids to send to each process
	std::vector< size_type > before_ids_count( nProcs , 0 );
	for(size_type i=0;i<localElementCountBefore;i++)
	{
	    int process_for_id = get_process_for_id( localIdsBefore[i], nProcs, allIdMin, allIdMax );
        assert( localIdsBefore[i]>=allIdMin && localIdsBefore[i]<allIdMax );
        assert( id_in_process_id_range( localIdsBefore[i], allIdMin, allIdMax, process_for_id, nProcs) );
	    ++ before_ids_count[ process_for_id ];
	}

    // count how many ids to receive from each process
    std::vector< size_type > after_ids_count( nProcs , 0 );
	for(size_type i=0;i<localElementCountAfter;i++)
	{
        int process_for_id = get_process_for_id( localIdsAfter[i], nProcs, allIdMin, allIdMax );
        assert( localIdsAfter[i]>=allIdMin && localIdsAfter[i]<allIdMax );
        assert( id_in_process_id_range( localIdsAfter[i], allIdMin, allIdMax, process_for_id, nProcs) );
        ++ after_ids_count[ process_for_id ];
	}

    // **** communication sizes computation ****
    send_count.resize( nProcs, -1 );
    for(int i=0;i<nProcs;i++)
    {
        send_count[i] = IdLocalizationSerializeBuffer::bufferSize( before_ids_count[i] , after_ids_count[i] ) ;
    }
    recv_count.resize( nProcs, -1 );
    MPI_Alltoall( send_count.data() , 1 , MPI_INT , recv_count.data() , 1 , MPI_INT , comm );

    // **** communication buffer allocation ****
    size_type totalSendSize = 0;
    size_type totalRecvSize = 0;
    for(int i=0;i<nProcs;i++)
    {
        assert( recv_count[i]>=0 );
        totalSendSize += send_count[i];
        totalRecvSize += recv_count[i];
    }
    std::vector<char> comm_buffer( totalSendSize + totalRecvSize );

    // fill in send buffer
    std::vector< size_type > serialize_id_counter( nProcs );
    std::vector< IdLocalizationSerializeBuffer* > serialize_buffers( nProcs , nullptr );
    char* comm_buffer_ptr = comm_buffer.data();
    for(int i=0;i<nProcs;i++)
    {
        serialize_buffers[i] = IdLocalizationSerializeBuffer::alloc( before_ids_count[i] , after_ids_count[i] , comm_buffer_ptr );
        comm_buffer_ptr += serialize_buffers[i]->size();
    }
    // at this point, comm_buffer_ptr points to the begining of the receive area of comm_buffer


    // "Before Ids" serialization
    for(int i=0;i<nProcs;i++) { serialize_id_counter[i] = 0; }
    for(index_type i=0;i<localElementCountBefore;i++)
    {
      id_type local_id = localIdsBefore[i];
      int process_for_id = get_process_for_id( local_id, nProcs, allIdMin, allIdMax );
      assert(process_for_id>=0 && process_for_id<nProcs);
      assert( id_in_process_id_range( local_id, allIdMin, allIdMax, process_for_id, nProcs) );
      index_type id_counter = serialize_id_counter[process_for_id] ++ ;
      assert( id_counter>=0 && id_counter<serialize_buffers[process_for_id]->beforeCount() );
      serialize_buffers[ process_for_id ]->beforeId( id_counter ).m_id = local_id;
      serialize_buffers[ process_for_id ]->beforeId( id_counter ).m_owner = rank;
      serialize_buffers[ process_for_id ]->beforeId( id_counter ).m_index = i;
    }

#   ifndef NDEBUG
    for(int i=0;i<nProcs;i++)
    {
      assert( serialize_id_counter[i] == serialize_buffers[i]->beforeCount() && serialize_id_counter[i] == before_ids_count[i] );
    }
#   endif

    // "After Ids" serialization
    for(int i=0;i<nProcs;i++) { serialize_id_counter[i] = 0; }
    for(index_type i=0;i<localElementCountAfter;i++)
    {
      id_type local_id = localIdsAfter[i];
      int process_for_id = get_process_for_id( local_id, nProcs, allIdMin, allIdMax );
      assert(process_for_id>=0 && process_for_id<nProcs);
      assert( id_in_process_id_range( local_id, allIdMin, allIdMax, process_for_id, nProcs) );
      index_type id_counter = serialize_id_counter[process_for_id] ++ ;
      assert( id_counter>=0 && id_counter<serialize_buffers[process_for_id]->afterCount() );
      serialize_buffers[ process_for_id ]->afterId( id_counter ).m_id = local_id;
      serialize_buffers[ process_for_id ]->afterId( id_counter ).m_owner = rank;
      serialize_buffers[ process_for_id ]->afterId( id_counter ).m_index = i;
    }

#   ifndef NDEBUG
    for(int i=0;i<nProcs;i++)
    {
      assert( serialize_id_counter[i] == serialize_buffers[i]->afterCount() && serialize_id_counter[i] == after_ids_count[i] );
    }
#   endif

    // compute displacements from counts, needed by alltoallv
    send_displ.resize( nProcs , -1 );
    recv_displ.resize( nProcs , -1 );
    send_displ[0] = recv_displ[0] = 0;
    for(int i=1;i<nProcs;i++)
    {
        send_displ[i] = send_displ[i-1] + send_count[i-1];
        recv_displ[i] = recv_displ[i-1] + recv_count[i-1];
    }

#   ifdef VERBOSE_DEBUG
    for(int i=0;i<nProcs;i++)
    {
      std::cout<<"P"<<rank<<": Sends to P"<<i<<": beforeCount="<<serialize_buffers[i]->beforeCount()<<", afterCount="<<serialize_buffers[i]->afterCount()
               <<", send count/displ="<<send_count[i]<<"/"<<send_displ[i]<<", recv count/displ="<<recv_count[i]<<"/"<<recv_displ[i]<<std::endl; std::cout.flush();
    }
#   endif

    // send Ids lists (before and after) to processes depending on the range affected to each
    MPI_Alltoallv( comm_buffer.data() , send_count.data(), send_displ.data() , MPI_CHAR , comm_buffer_ptr , recv_count.data() , recv_displ.data() , MPI_CHAR , comm );

    
    // read received infos
    //const size_type id_range_start = get_id_range_start_for_process(rank,nProcs,allIdMin,allIdMax);
    const size_type id_range_size = get_id_range_size_for_process(rank,nProcs,allIdMin,allIdMax);
    std::vector<IdMove> id_move( id_range_size , IdMove{0,0,-1,-1} );
    for(int i=0;i<nProcs;i++)
    {
        IdLocalizationSerializeBuffer* input = IdLocalizationSerializeBuffer::bindTo( comm_buffer_ptr );
        comm_buffer_ptr += input->size();
        size_type bcount = input->beforeCount();
#   	ifdef VERBOSE_DEBUG
          std::cout<<"P"<<rank<<": Receives from P"<<i<<": beforeCount="<<input->beforeCount()<<", afterCount="<<input->afterCount()<<std::endl; std::cout.flush();	
#	    endif
        for(index_type j=0;j<bcount;j++)
        {
            const IdLocalization& idloc = input->beforeId(j);
            id_type id = idloc.m_id;
            assert( id_in_process_id_range( id, allIdMin, allIdMax, rank, nProcs) );
            int owner = idloc.m_owner;
            index_type index = id - get_id_range_start_for_process( rank, nProcs, allIdMin, allIdMax );
            assert( owner>=0 && owner<nProcs );
            assert(index>=0 && index<id_move.size());
            assert( id_move[index].m_src_owner == -1 ); // every Id has at most one origin
            id_move[index].m_src_owner = owner;
            id_move[index].m_src_index = idloc.m_index;
        }
        size_type acount = input->afterCount();
        for(index_type j=0;j<acount;j++)
        {
            const IdLocalization& idloc = input->afterId(j);
            id_type id = idloc.m_id;
            assert( id_in_process_id_range( id, allIdMin, allIdMax, rank, nProcs) );
            int owner = idloc.m_owner;
            index_type index = id - get_id_range_start_for_process( rank, nProcs, allIdMin, allIdMax );
            assert( owner>=0 && owner<nProcs );
            assert(index>=0 && index<id_move.size());
            assert( id_move[index].m_dst_owner == -1 ); // every Id has at most one destination
            id_move[index].m_dst_owner = owner;
            id_move[index].m_dst_index = idloc.m_index;
        }
    }

#   ifndef NDEBUG
    for(index_type i=0;i<id_range_size;i++)
    {
        // ensures no ids are created ex-nihilo (we wouldn't know where to copy data from for this one )
        // element deletion is ok
        assert( id_move[i].m_src_owner!=-1 && id_move[i].m_dst_owner!=-1 );
#       if VERBOSE_DEBUG >= 2
        if( id_move[i].m_src_owner==-1 || id_move[i].m_dst_owner==-1 )
        {
            std::cout<<"P"<<rank<<": Id #"<< (i+id_range_start) << " : P"<<id_move[i].m_src_owner<<" => P"<<id_move[i].m_dst_owner << std::endl; std::cout.flush();
        }
        if( id_move[i].m_src_owner != id_move[i].m_dst_owner )
        {
            std::cout<<"P"<<rank<<": Id #"<< (i+id_range_start) << " : P"<<id_move[i].m_src_owner<<" => P"<<id_move[i].m_dst_owner << std::endl; std::cout.flush();
        }
#       endif
    }
#   endif

    //size_type id_move_cursor = 0;
    for(int i=0;i<nProcs;i++) { serialize_id_counter[i] = 0; }
    for(index_type i=0;i<id_range_size;i++)
    {
        ++ serialize_id_counter[ id_move[i].m_src_owner ];
    }

    // resolve recv counts for future Alltoallv
    for(int i=0;i<nProcs;i++)
    {
        send_count[i] = IdMoveSerializeBuffer::bufferSize( serialize_id_counter[i]  ) ;
    }
    MPI_Alltoall( send_count.data() , 1 , MPI_INT , recv_count.data() , 1 , MPI_INT , comm );

    // fill in send buffer
    size_type totalMoveBufferSize = 0;
    for(int i=0;i<nProcs;i++)
    {
      totalMoveBufferSize += send_count[i] ;
      totalMoveBufferSize += recv_count[i] ;
    }

    // allocate serialization buffer and initialize pointers for each target processor
    comm_buffer.resize( totalMoveBufferSize );
    comm_buffer_ptr = comm_buffer.data();
    std::vector< IdMoveSerializeBuffer* > move_serialize_buffers( nProcs , nullptr );
    for(int i=0;i<nProcs;i++)
    {
        move_serialize_buffers[i] = IdMoveSerializeBuffer::alloc( serialize_id_counter[i] , comm_buffer_ptr );
        comm_buffer_ptr += move_serialize_buffers[i]->size();
    }
    // at this point, comm_buffer_ptr points to the receive area of comm_buffer

    // serialize data to send
    for(int i=0;i<nProcs;i++) { serialize_id_counter[i] = 0; }
    for(index_type i=0;i<id_range_size;i++)
    {
        int src_owner = id_move[i].m_src_owner;
        index_type move_offset = serialize_id_counter[ id_move[i].m_src_owner ] ++ ;
        move_serialize_buffers[ src_owner ]->id_move(move_offset) = id_move[i]; // a condenser
    }

#   ifndef NDEBUG
    for(int i=0;i<nProcs;i++)
    {
      assert( move_serialize_buffers[i]->count() == serialize_id_counter[i] );
    }
#   endif

    // compute displacements from counts, needed by alltoallv
    send_displ[0] = recv_displ[0] = 0;
    for(int i=1;i<nProcs;i++)
    {
        send_displ[i] = send_displ[i-1] + send_count[i-1];
        recv_displ[i] = recv_displ[i-1] + recv_count[i-1];
    }
    
    // send partial movement lists to each processors
    MPI_Alltoallv( comm_buffer.data() , send_count.data(), send_displ.data() , MPI_CHAR , comm_buffer_ptr , recv_count.data() , recv_displ.data() , MPI_CHAR , comm );
    
    // read movement lists and build send/receive indices
    [[maybe_unused]] size_type send_indices_count = 0;
    
    for(int i=0;i<nProcs;i++) { send_count[i] = 0; }
    char* receive_buffer_ptr = comm_buffer_ptr; // receive_buffer_ptr stores the point in comm_buffer where the receive area is
    for(int i=0;i<nProcs;i++)
    {
        IdMoveSerializeBuffer* input = IdMoveSerializeBuffer::bindTo( comm_buffer_ptr );
        comm_buffer_ptr += input->size();
        size_type count = input->count();
        for(index_type j=0;j<count;j++)
        {
          const IdMove& idmove = input->id_move(j);
          assert( idmove.m_src_owner == rank );
          assert( idmove.m_dst_owner>=0 && idmove.m_dst_owner<nProcs );
#         if VERBOSE_DEBUG >= 2
            std::cout <<"P"<<rank<<": send "<<idmove.m_src_index<<" -> P"<<idmove.m_dst_owner<<" @"<<idmove.m_dst_index<<std::endl;
            std::cout.flush();
#         endif
          ++ send_count[idmove.m_dst_owner];
          ++ send_indices_count;
        }
    }
    assert( send_indices_count == localElementCountBefore );

    MPI_Alltoall( send_count.data() , 1 , MPI_INT , recv_count.data() , 1 , MPI_INT , comm );

    [[maybe_unused]] size_type recv_indices_count = 0;
    for(int i=0;i<nProcs;i++)
    {
      recv_indices_count += recv_count[i];
    }
    assert( recv_indices_count == localElementCountAfter );

    // compute displacements from counts
    send_displ[0] = 0;
    for(int i=1;i<nProcs;i++)
    {
        send_displ[i] = send_displ[i-1] + send_count[i-1];
    }
  
    send_indices.resize( localElementCountBefore, -1 );
    std::vector<int> partial_recv_indices( localElementCountBefore, -1 );
    for(int i=0;i<nProcs;i++) { serialize_id_counter[i] = 0; }
    comm_buffer_ptr = receive_buffer_ptr; // restore comm_buffer_ptr to the start of the receive area of comm_buffer;
    for(int i=0;i<nProcs;i++)
    {
        IdMoveSerializeBuffer* input = IdMoveSerializeBuffer::bindTo( comm_buffer_ptr );
        comm_buffer_ptr += input->size();
        size_type count = input->count();
        for(index_type j=0;j<count;j++)
        {
          const IdMove& idmove = input->id_move(j);
          unsigned int send_buffer_index = send_displ[idmove.m_dst_owner] + serialize_id_counter[idmove.m_dst_owner];
          assert( send_buffer_index>=0 && send_buffer_index<localElementCountBefore );
          partial_recv_indices[ send_buffer_index ] = idmove.m_dst_index;
          send_indices[ send_buffer_index ] = idmove.m_src_index;
          ++ serialize_id_counter[idmove.m_dst_owner];
        }
    }

#   ifndef NDEBUG
    for(unsigned int i=0;i<localElementCountBefore;i++)
    {
      assert( partial_recv_indices[i]>=0 );
      assert( send_indices[i]>=0 && static_cast<size_type>(send_indices[i])<localElementCountBefore );
    }
#   endif

    // compute displacements from counts
    recv_displ[0] = 0;
    for(int i=1;i<nProcs;i++)
    {
        recv_displ[i] = recv_displ[i-1] + recv_count[i-1];
    }
    assert( recv_indices_count == localElementCountAfter );

    // send partial movement lists to each processors
    recv_indices.resize( localElementCountAfter, -1 );
    MPI_Alltoallv( partial_recv_indices.data() , send_count.data(), send_displ.data() , MPI_INT , recv_indices.data() , recv_count.data() , recv_displ.data() , MPI_INT , comm );

#   ifndef DEBUG
    for(size_type i=0;i<localElementCountAfter;i++)
    {
      assert( recv_indices[i]>=0 && static_cast<size_type>(recv_indices[i])<localElementCountAfter );
    }
#   endif

    // finally, convert send_indices and receive_indices from gather indirection tables to scatter indirection table
    {
      std::vector<int> tmp = send_indices;
      for(size_type i=0;i<localElementCountBefore;i++)
      {
        assert( tmp[i]>=0 && static_cast<size_type>(tmp[i])<localElementCountBefore );
        send_indices[tmp[i]] = i;
      }
      tmp = recv_indices;
      for(size_type i=0;i<localElementCountAfter;i++)
      {
        assert( tmp[i]>=0 && static_cast<size_type>(tmp[i])<localElementCountAfter );
        recv_indices[tmp[i]] = i;
      }
    }

    // final result : send_indices, send_count, send_displ, recv_indices, recv_count, recv_displ
#   ifdef VERBOSE_DEBUG
    send_indices_count = 0;
    std::cout<<"P"<<rank<<": send indices = ";
    for(int i=0;i<nProcs;i++) 
    {
      int n = send_count[i];
      std::cout<<"P"<<i<<"=[ ";
      for(int j=0;j<n;j++)
      {
        std::cout<<send_indices[ send_indices_count++ ]<<" ";
      }
      std::cout<<"] ";
    }
    std::cout<< std::endl; std::cout.flush();
    assert( send_indices_count == localElementCountBefore );

    std::cout<<"P"<<rank<<": recv indices = ";
    recv_indices_count = 0;
    for(int i=0;i<nProcs;i++) 
    {
      int n = recv_count[i];
      std::cout<<"P"<<i<<"=[ ";
      for(int j=0;j<n;j++)
      {
        std::cout<<recv_indices[ recv_indices_count++ ]<<" ";
      }
      std::cout<<"] ";
    }
    std::cout<< std::endl; std::cout.flush();
    assert( recv_indices_count == localElementCountAfter );
#   endif

}

} // namespace XsDataMove
