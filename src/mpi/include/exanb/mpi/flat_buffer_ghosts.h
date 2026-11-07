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

#pragma once

#include <exanb/mpi/ghosts_comm_scheme.h>
#include <onika/log.h>

#include <mpi.h>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace exanb
{

  /*
   * A per-particle array of rows, in grid flat order (row of a particle = cell_particle_offset[cell] + index in cell,
   * ghost cells included), owned by someone else (a potential's message buffers, for instance). Rows hold
   * m_row_size values of type float or double, m_row_stride values apart.
   * ghost_update_flat_buffer copies owner rows to their ghost images, update_flat_buffer_from_ghost sums ghost rows
   * into their owners, following the ghost communication scheme of the last ghost update (same particles, same order).
   * Unlike ghost_update_opt / update_opt_from_ghost (grid-owned optional fields of type double, Vec3d or Mat3d),
   * rows can be any number of float or double values, the buffer stays owned by the caller, and
   * flat_buffer_ghost_exchange can be called directly from C++, several times inside one operator.
   */
  struct GhostFlatBuffer
  {
    enum class ValueType : int { Float = 0, Double = 1 };
    void * m_data = nullptr;
    size_t m_row_size = 0;
    size_t m_row_stride = 0;
    size_t m_nrows = 0;
    ValueType m_type = ValueType::Float;

    inline size_t value_bytes() const { return m_type == ValueType::Float ? sizeof(float) : sizeof(double); }
    inline size_t row_bytes() const { return m_row_size * value_bytes(); }
    inline uint8_t * row(size_t i) const { return static_cast<uint8_t*>(m_data) + i * m_row_stride * value_bytes(); }
  };

  namespace flat_buffer_ghosts_details
  {
    template<class T>
    static inline void add_row( uint8_t * dst , const uint8_t * src , size_t n )
    {
      T * d = reinterpret_cast<T*>(dst);
      const T * s = reinterpret_cast<const T*>(src);
      for(size_t k=0;k<n;k++) d[k] += s[k];
    }
    static inline void add_row( const GhostFlatBuffer& b , uint8_t * dst , const uint8_t * src )
    {
      if( b.m_type == GhostFlatBuffer::ValueType::Float ) add_row<float>( dst, src, b.m_row_size );
      else add_row<double>( dst, src, b.m_row_size );
    }
    static inline int mpi_count( size_t bytes )
    {
      if( bytes > size_t( std::numeric_limits<int>::max() ) ) onika::fatal_error() << "flat buffer ghost message of " << bytes << " bytes exceeds MPI int count" << std::endl;
      return int( bytes );
    }
  }

  // ToGhost = true : owner rows -> ghost rows (copy). false : ghost rows -> owner rows (sum).
  template<bool ToGhost>
  static inline void flat_buffer_ghost_exchange( MPI_Comm comm , const GhostCommunicationScheme& scheme , const size_t * cell_particle_offset , const GhostFlatBuffer& b , int tag )
  {
    using namespace flat_buffer_ghosts_details;
    int nprocs = 1, rank = 0;
    MPI_Comm_size( comm, &nprocs );
    MPI_Comm_rank( comm, &rank );
    const size_t rb = b.row_bytes();
    if( rb == 0 || scheme.m_partner.empty() ) return;
    assert( int(scheme.m_partner.size()) == nprocs );

    // a send side (owner rows, listed by the sends) and a receive side (ghost rows, listed by the receives) per partner;
    // the message layout is the same on both sides: send cell k <-> receive cell k, same particle order
    auto owner_rows = [&](const GhostPartnerCommunicationScheme& ps, auto func)
    {
      const size_t ncells = ps.m_sends.size();
#     pragma omp parallel for schedule(dynamic) if( ToGhost )
      for(size_t k=0;k<ncells;k++)
      {
        const auto& s = ps.m_sends[k];
        const size_t base = cell_particle_offset[ s.m_cell_i ];
        const size_t np = s.m_particle_i.size();
        for(size_t j=0;j<np;j++) func( base + s.m_particle_i[j] , s.m_send_buffer_offset + j );
      }
    };
    auto ghost_rows = [&](const GhostPartnerCommunicationScheme& ps, auto func)
    {
      const size_t ncells = ps.m_receives.size();
#     pragma omp parallel for schedule(dynamic)
      for(size_t k=0;k<ncells;k++)
      {
        const auto info = ghost_cell_receive_info( ps.m_receives[k] );
        const size_t base = cell_particle_offset[ info.m_cell_i ];
        for(size_t j=0;j<info.m_n_particles;j++) func( base + j , ps.m_receive_offset[k] + j );
      }
    };

    std::vector< std::vector<uint8_t> > sendbuf( nprocs ), recvbuf( nprocs );
    std::vector<MPI_Request> requests;
    std::vector<int> request_partner;

    // receives first
    for(int p=0;p<nprocs;p++)
    {
      if( p == rank ) continue;
      const auto& ps = scheme.m_partner[p];
      const size_t nrecv = ToGhost ? ps.m_particles_to_receive : ps.m_particles_to_send;
      if( nrecv == 0 ) continue;
      recvbuf[p].resize( nrecv * rb );
      requests.push_back( MPI_REQUEST_NULL );
      request_partner.push_back( p );
      MPI_Irecv( recvbuf[p].data(), mpi_count(recvbuf[p].size()), MPI_BYTE, p, tag, comm, &requests.back() );
    }

    // pack and send
    const size_t first_send = requests.size();
    for(int p=0;p<nprocs;p++)
    {
      const auto& ps = scheme.m_partner[p];
      const size_t nsend = ToGhost ? ps.m_particles_to_send : ps.m_particles_to_receive;
      if( nsend == 0 ) continue;
      auto& buf = sendbuf[p];
      buf.resize( nsend * rb );
      uint8_t * out = buf.data();
      if constexpr ( ToGhost ) owner_rows( ps , [&](size_t row, size_t slot) { std::memcpy( out + slot*rb , b.row(row) , rb ); } );
      else ghost_rows( ps , [&](size_t row, size_t slot) { std::memcpy( out + slot*rb , b.row(row) , rb ); } );
      if( p != rank )
      {
        requests.push_back( MPI_REQUEST_NULL );
        request_partner.push_back( -1 );
        MPI_Isend( buf.data(), mpi_count(buf.size()), MPI_BYTE, p, tag, comm, &requests.back() );
      }
    }

    // unpack one partner's message
    auto unpack = [&](int p, const std::vector<uint8_t>& buf)
    {
      const auto& ps = scheme.m_partner[p];
      const uint8_t * in = buf.data();
      if constexpr ( ToGhost )
      {
        ghost_rows( ps , [&](size_t row, size_t slot) { std::memcpy( b.row(row) , in + slot*rb , rb ); } );
      }
      else
      {
        // an owner may have several ghost images: sequential sum (owner_rows is not parallel in this direction)
        owner_rows( ps , [&](size_t row, size_t slot) { add_row( b, b.row(row), in + slot*rb ); } );
      }
    };

    // loopback (periodic images within this process): pack buffer read directly
    if( size_t(rank) < scheme.m_partner.size() )
    {
      const auto& ps = scheme.m_partner[rank];
      if( ps.m_particles_to_send > 0 || ps.m_particles_to_receive > 0 ) unpack( rank , sendbuf[rank] );
    }

    // remote messages, unpacked as they arrive
    size_t remaining = first_send;
    while( remaining > 0 )
    {
      int idx = MPI_UNDEFINED;
      MPI_Waitany( int(first_send), requests.data(), &idx, MPI_STATUS_IGNORE );
      if( idx == MPI_UNDEFINED ) break;
      unpack( request_partner[idx] , recvbuf[ request_partner[idx] ] );
      -- remaining;
    }
    if( requests.size() > first_send )
    {
      MPI_Waitall( int(requests.size() - first_send), requests.data() + first_send, MPI_STATUSES_IGNORE );
    }
  }

}
