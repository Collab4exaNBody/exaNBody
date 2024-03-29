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

#include <vector>
#include <cstdlib>
#include <unordered_map>

#include <exanb/core/particle_id_codec.h>
#include <exanb/core/basic_types.h>
#include <onika/memory/allocator.h> 
#include <onika/cuda/cuda.h> 

#include <mpi.h>

namespace exanb
{
  

  struct GhostCellSendScheme
  {
    size_t m_cell_i;  // sender's cell index
    size_t m_partner_cell_i;  // partner's cell index
    ssize_t m_send_buffer_offset = -1;
    double m_x_shift;
    double m_y_shift;
    double m_z_shift;
    onika::memory::CudaMMVector<uint32_t> m_particle_i; // sender's particle index

    GhostCellSendScheme() = default;
    GhostCellSendScheme(const GhostCellSendScheme&) = default;

    inline GhostCellSendScheme( GhostCellSendScheme && other )
      : m_cell_i( other.m_cell_i )
      , m_partner_cell_i( other.m_partner_cell_i )
      , m_x_shift( other.m_x_shift )
      , m_y_shift( other.m_y_shift )
      , m_z_shift( other.m_z_shift )
      , m_particle_i( std::move(other.m_particle_i) ) {}

    inline GhostCellSendScheme& operator = ( GhostCellSendScheme && other )
    {
      m_cell_i = other.m_cell_i;
      m_partner_cell_i = other.m_partner_cell_i;
      m_x_shift = other.m_x_shift;
      m_y_shift = other.m_y_shift;
      m_z_shift = other.m_z_shift;
      m_particle_i = std::move(other.m_particle_i);
      return *this;
    }
  };

  struct GhostCellReceiveSchemeDetail
  {
    size_t m_cell_i; // receiver's cell index where to insert particles
    size_t m_n_particles; // number of particles updated from original cell
  };

  using GhostCellReceiveScheme = uint64_t;
  
  ONIKA_HOST_DEVICE_FUNC
  static inline GhostCellReceiveSchemeDetail ghost_cell_receive_info( uint64_t r )
  {
    size_t cell_i=0;
    size_t n_particles=0;
    decode_cell_particle(r, cell_i , n_particles);
    return { cell_i , n_particles };
  }

  struct GhostSendCellInfo
  {
    uint32_t m_dst;            // destination process
    uint32_t m_send_cell_i;   // send cell index (index in GhostPartnerCommunicationScheme::m_sends)
    size_t m_send_buffer_offset; // where to place particles in corresponding send buffer
  };

  // WARNING: m_cell_counter is undefined after a copy or move copy construction
  struct GhostPartnerSendBuffer
  {
    int64_t m_cells_to_send; // to reset m_cell_counter after it has been decremented to 0
    std::atomic<int64_t> m_cell_counter;
    std::vector<uint8_t> m_buffer;
    GhostPartnerSendBuffer() = default;
    inline GhostPartnerSendBuffer(const GhostPartnerSendBuffer& other) : m_buffer(other.m_buffer) {}
    inline GhostPartnerSendBuffer(const GhostPartnerSendBuffer&& other) : m_buffer(std::move(other.m_buffer)) {}
  };

  struct GhostCommSendBuffers
  {
    std::unordered_multimap<IJK,GhostSendCellInfo> m_send_cell_info;
    std::vector<GhostPartnerSendBuffer> m_send_buffer;
    std::vector< MPI_Request > m_send_requests;
  };  
  
  struct GhostCommReceiveBuffers
  {
    std::vector< std::vector<uint8_t> > m_receive_buffers;
    std::vector< MPI_Request > m_receive_requests;
  };

  struct GhostPartnerCommunicationScheme
  {
    onika::memory::CudaMMVector< GhostCellSendScheme > m_sends;
    onika::memory::CudaMMVector< GhostCellReceiveScheme > m_receives;
    onika::memory::CudaMMVector< uint64_t > m_receive_offset;
  };

  struct GhostCommunicationScheme
  {
    IJK m_grid_dims = { 0, 0, 0 };
    size_t m_particle_bytes = 0;
    size_t m_cell_bytes = 0;
    std::vector< GhostPartnerCommunicationScheme > m_partner;
  };

  template<class StreamT>
  static inline StreamT& to_stream( StreamT& out , const GhostCommunicationScheme& cs)
  {
    for(size_t p=0;p<cs.m_partner.size();p++)
    {
      out << "partner "<<p<<" :"<<std::endl;
      out << "  sends :"<<std::endl;
      for(size_t c=0;c<cs.m_partner[p].m_sends.size();c++)
      {
        out << "    cell #"<<cs.m_partner[p].m_sends[c].m_cell_i<<" to cell #"<<cs.m_partner[p].m_sends[c].m_partner_cell_i<<" with shift "
            <<cs.m_partner[p].m_sends[c].m_x_shift<<","<<cs.m_partner[p].m_sends[c].m_y_shift<<","<<cs.m_partner[p].m_sends[c].m_z_shift<<std::endl;
      }
      out << "  receives :"<<std::endl;
      for(size_t c=0;c<cs.m_partner[p].m_receives.size();c++)
      {
        auto rinfo = ghost_cell_receive_info(cs.m_partner[p].m_receives[c]);
        out<<"    cell #"<<rinfo.m_cell_i<<" containing "<<rinfo.m_n_particles<<std::endl;
      }
    }
    return out;
  }

}

