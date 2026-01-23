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
#include <onika/math/basic_types.h>
#include <onika/memory/allocator.h> 
#include <onika/cuda/cuda.h> 
#include <onika/cuda/cuda_math.h> 

#include <mpi.h>

namespace exanb
{

  // ghost particle coordinate filter,
  // apply periodic or mirror boundary conditions to one ore more of the particle's coordinate
  struct GhostBoundaryModifier
  {
    static inline constexpr uint32_t SHIFT_X        = 1u << 0;
    static inline constexpr uint32_t MIRROR_X       = 1u << 1;
    static inline constexpr uint32_t SIDE_X         = 1u << 2;
    static inline constexpr uint32_t MASK_SHIFT_X   = 0;
    
    static inline constexpr uint32_t SHIFT_Y        = 1u << 3;
    static inline constexpr uint32_t MIRROR_Y       = 1u << 4;
    static inline constexpr uint32_t SIDE_Y         = 1u << 5;
    static inline constexpr uint32_t MASK_SHIFT_Y   = 3;
    
    static inline constexpr uint32_t SHIFT_Z        = 1u << 6;
    static inline constexpr uint32_t MIRROR_Z       = 1u << 7;
    static inline constexpr uint32_t SIDE_Z         = 1u << 8;
    static inline constexpr uint32_t MASK_SHIFT_Z   = 6;
    
    Vec3d m_domain_min = { 0. , 0. , 0. };
    Vec3d m_domain_max = { 0. , 0. , 0. };

    ONIKA_HOST_DEVICE_FUNC
    static inline bool bit(uint32_t flags, uint32_t mask) { return ( flags & mask ) == mask ; }

    ONIKA_HOST_DEVICE_FUNC
    static inline double apply_vector_modifier(double x, uint32_t flags) // velocity or force
    {
      return x * ( bit(flags,MIRROR_X) ? -1.0 : 1.0 );
    }
    
    ONIKA_HOST_DEVICE_FUNC
    static inline double apply_coord_modifier(double x, double rmin, double rmax, uint32_t flags)
    {
           if( bit(flags, SHIFT_X) ) return  x + ( bit(flags,SIDE_X) ? 1.0  : -1.0 ) * ( rmax - rmin );
      else if( bit(flags,MIRROR_X) ) return -x + ( bit(flags,SIDE_X) ? rmax : rmin ) * 2.0 ;
      else return x;
    }

    ONIKA_HOST_DEVICE_FUNC
    inline double apply_rx_modifier(double rx, uint32_t flags) const
    {
      return apply_coord_modifier( rx , m_domain_min.x , m_domain_max.x , flags >> MASK_SHIFT_X );
    }

    ONIKA_HOST_DEVICE_FUNC
    inline double apply_ry_modifier(double ry, uint32_t flags) const
    {
      return apply_coord_modifier( ry , m_domain_min.y , m_domain_max.y , flags >> MASK_SHIFT_Y );
    }

    ONIKA_HOST_DEVICE_FUNC
    inline double apply_rz_modifier(double rz, uint32_t flags) const
    {
      return apply_coord_modifier( rz , m_domain_min.z , m_domain_max.z , flags >> MASK_SHIFT_Z );
    }

    ONIKA_HOST_DEVICE_FUNC
    inline Vec3d apply_r_modifier(const Vec3d& r , uint32_t flags) const
    {
      return { apply_rx_modifier(r.x,flags) , apply_ry_modifier(r.y,flags) , apply_rz_modifier(r.z,flags) }; 
    }
    
    template<class StreamT>
    static inline void print_flags(StreamT& out, uint32_t flags)
    {
      out<<(bit(flags,SHIFT_X)?" SHIFT_X":"")<<(bit(flags,MIRROR_X)?" MIRROR_X":"")<<(bit(flags,SIDE_X)?" SIDE_X":"") 
         <<(bit(flags,SHIFT_Y)?" SHIFT_Y":"")<<(bit(flags,MIRROR_Y)?" MIRROR_Y":"")<<(bit(flags,SIDE_Y)?" SIDE_Y":"") 
         <<(bit(flags,SHIFT_Z)?" SHIFT_Z":"")<<(bit(flags,MIRROR_Z)?" MIRROR_Z":"")<<(bit(flags,SIDE_Z)?" SIDE_Z":"") ;
    }
  };

  struct GhostCellSendScheme
  {
    size_t m_cell_i = 0;  // sender's cell index
    size_t m_partner_cell_i = 0;  // partner's cell index
    ssize_t m_send_buffer_offset = -1;
    uint32_t m_flags = 0; // GhostBoundaryModifier compatible flags to pilot coordinate filtering
    onika::memory::CudaMMVector<uint32_t> m_particle_i; // sender's particle index

    GhostCellSendScheme() = default;
    GhostCellSendScheme(const GhostCellSendScheme&) = default;

    inline GhostCellSendScheme( GhostCellSendScheme && other )
      : m_cell_i( other.m_cell_i )
      , m_partner_cell_i( other.m_partner_cell_i )
      , m_flags( other.m_flags )
      , m_particle_i( std::move(other.m_particle_i) ) {}

    inline GhostCellSendScheme& operator = ( GhostCellSendScheme && other )
    {
      m_cell_i = other.m_cell_i;
      m_partner_cell_i = other.m_partner_cell_i;
      m_flags = other.m_flags;
      m_particle_i = std::move(other.m_particle_i);
      return *this;
    }
  };

  struct GhostCellReceiveSchemeDetail
  {
    size_t m_cell_i;      // receiver's cell index where to insert particles
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
    uint32_t m_dst;              // destination process
    uint32_t m_send_cell_i;      // send cell index (index in GhostPartnerCommunicationScheme::m_sends)
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

    // control values, can be be rebuilt from previous containers.
    // they are computed and stored here for 2 reasons :
    // 1. to check consistency of data for debug pruposes
    // 2. allow for communication buffer resize without touching memory in CudaMMVector, avoiding back-and-forth CPU/GPU memory moves
    size_t m_particles_to_receive = 0;
    size_t m_particles_to_send = 0;
  };

  struct GhostCommunicationScheme
  {
    // Communication bject's unique id :
    // two identical communication mays have different uids,
    // but two different communication objects cannot have the same uid
    // this might be use to cache computations of structures/values derived from this object
    int64_t m_uid = 0;
    
    IJK m_grid_dims = { 0, 0, 0 };
    size_t m_particle_bytes = 0;
    size_t m_cell_bytes = 0;
    std::vector< GhostPartnerCommunicationScheme > m_partner;

    static inline std::atomic<int64_t> s_uid_counter = 0;
    static inline int64_t generate_uid() { return s_uid_counter.fetch_add( 1 ); }
    inline void update_uid()
    {
      m_uid = generate_uid();
    }
    inline int64_t uid() const { return m_uid; }
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
        out << "    cell #"<<cs.m_partner[p].m_sends[c].m_cell_i<<" to cell #"<<cs.m_partner[p].m_sends[c].m_partner_cell_i<<" with flags "
            << std::hex << cs.m_partner[p].m_sends[c].m_flags << std::dec << std::endl;
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

