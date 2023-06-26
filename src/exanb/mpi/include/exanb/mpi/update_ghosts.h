#pragma once

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/log.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/grid.h>
#include <exanb/grid_cell_particles/grid_cell_values.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/particle_id_codec.h>
#include <exanb/field_sets.h>
#include <exanb/core/check_particles_inside_cell.h>

#include <onika/soatl/field_tuple.h>

#include <vector>
#include <string>
#include <list>
#include <algorithm>
#include <tuple>

#include <mpi.h>
#include <exanb/mpi/update_ghost_utils.h>
#include <exanb/mpi/ghosts_comm_scheme.h>
#include <exanb/mpi/data_types.h>

namespace exanb
{
  
  using namespace UpdateGhostsUtils;

  template<
    class GridT,
    class FieldSetT,
    bool CreateParticles,
    class = AssertGridContainFieldSet<GridT,FieldSetT>
    >
  struct UpdateGhostsNode : public OperatorNode
  {  
    using CellParticles = typename GridT::CellParticles;
    using ParticleFullTuple = typename CellParticles::TupleValueType;
    using ParticleTuple = typename FieldSetToParticleTuple< AddDefaultFields<FieldSetT> >::type;
    
    static_assert( ParticleTuple::has_field(field::rx) , "ParticleTuple must contain field::rx" );
    static_assert( ParticleTuple::has_field(field::ry) , "ParticleTuple must contain field::ry" );
    static_assert( ParticleTuple::has_field(field::rz) , "ParticleTuple must contain field::rz" );
    
    struct CellParticlesUpdateData
    {
      size_t m_cell_i;
      ParticleTuple m_particles[0];
    };
    static_assert( sizeof(CellParticlesUpdateData) == sizeof(size_t) , "Unexpected size for CellParticlesUpdateData");
    static_assert( sizeof(uint8_t) == 1 , "uint8_t is not a byte");

    // -----------------------------------------------
    // Operator slots
    // -----------------------------------------------
    ADD_SLOT( MPI_Comm                 , mpi               , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( GhostCommunicationScheme , ghost_comm_scheme , INPUT , REQUIRED );
    ADD_SLOT( GridT                    , grid              , INPUT_OUTPUT);
    ADD_SLOT( GridCellValues           , grid_cell_values  , INPUT_OUTPUT , OPTIONAL );
    ADD_SLOT( long                     , mpi_tag           , INPUT , 0 );

    inline void execute () override final
    {
      using GridCellValueType = typename GridCellValues::GridCellValueType;
      
      // prerequisites

      MPI_Comm comm = *mpi;
      GhostCommunicationScheme& comm_scheme = *ghost_comm_scheme;

      CellParticles* cells = grid->cells();

      // per cell scalar values, if any
      GridCellValueType* cell_scalars = nullptr;
      unsigned int cell_scalar_components = 0;
      if( grid_cell_values.has_value() )
      {
        cell_scalar_components = grid_cell_values->components();
        if( cell_scalar_components > 0 )
        {
          assert( grid_cell_values->data().size() == grid->number_of_cells() * cell_scalar_components );
          cell_scalars = grid_cell_values->data().data();
        }
      }
      if( cell_scalars )
      {
        ldbg << "update ghost cell values with "<< cell_scalar_components << " components"<<std::endl;
      }

#     ifndef NDEBUG
      const size_t n_cells = grid->number_of_cells();
#     endif      
      
      int comm_tag = *mpi_tag;
      int nprocs = 1;
      int rank = 0;
      MPI_Comm_size(comm,&nprocs);
      MPI_Comm_rank(comm,&rank);

      // initialize MPI requests for both sends and receives
      size_t total_requests = 2 * nprocs;
      std::vector< MPI_Request > requests( total_requests );
      for(size_t i=0;i<total_requests;i++) { requests[i] = MPI_REQUEST_NULL; }

      // send and receive buffers
      std::vector< std::vector<uint8_t> > send_buffer(nprocs);
      std::vector< std::vector<uint8_t> > receive_buffer(nprocs);

      size_t active_sends = 0;
      size_t active_recvs = 0;
      for(int p=0;p<nprocs;p++)
      {
        // start receive from partner p
        size_t cells_to_receive = comm_scheme.m_partner[p].m_receives.size();
        if( cells_to_receive > 0 )
        {
          size_t particles_to_receive = 0;
          for(size_t i=0;i<cells_to_receive;i++)
          {
            particles_to_receive += ghost_cell_receive_info(comm_scheme.m_partner[p].m_receives[i]).m_n_particles;
          }
          assert( particles_to_receive > 0 );
          size_t receive_size = ( cells_to_receive * sizeof(CellParticlesUpdateData) ) + ( particles_to_receive * sizeof(ParticleTuple) );
          if( cell_scalars != nullptr )
          {
            receive_size += cells_to_receive * sizeof(GridCellValueType) * cell_scalar_components;
          }
          receive_buffer[p].resize( receive_size );
          //ldbg << "receiving "<<receive_size<<" bytes from P"<<p<<std::endl;
          ++ active_recvs;
          MPI_Irecv( (char*) receive_buffer[p].data(), receive_size, MPI_CHAR, p, comm_tag, comm, & requests[p] );
        }
        
        // start send to partner p
        size_t cells_to_send = comm_scheme.m_partner[p].m_sends.size();
        if( cells_to_send > 0 )
        {
          size_t particles_to_send = 0;
          for(size_t i=0;i<cells_to_send;i++)
          {
            particles_to_send += comm_scheme.m_partner[p].m_sends[i].m_particle_i.size();
          }
          assert( particles_to_send > 0 );
          size_t send_buffer_size = ( cells_to_send * sizeof(CellParticlesUpdateData) ) + ( particles_to_send * sizeof(ParticleTuple) );
          if( cell_scalars != nullptr )
          {
            send_buffer_size += cells_to_send * sizeof(GridCellValueType) * cell_scalar_components;
          }
          send_buffer[p].resize( send_buffer_size );
          uint8_t* data_ptr = send_buffer[p].data();
          size_t data_cur = 0;
          CellParticlesUpdateData* data = reinterpret_cast<CellParticlesUpdateData*>( data_ptr+data_cur );
          for(size_t i=0;i<cells_to_send;i++)
          {
            assert(data_cur < send_buffer[p].size() );
            data->m_cell_i = comm_scheme.m_partner[p].m_sends[i].m_partner_cell_i;
            const size_t cell_i = comm_scheme.m_partner[p].m_sends[i].m_cell_i;
            const double rx_shift = comm_scheme.m_partner[p].m_sends[i].m_x_shift;
            const double ry_shift = comm_scheme.m_partner[p].m_sends[i].m_y_shift;
            const double rz_shift = comm_scheme.m_partner[p].m_sends[i].m_z_shift;
            assert( cell_i>=0 && cell_i<n_cells );
            uint32_t const * particle_index = comm_scheme.m_partner[p].m_sends[i].m_particle_i.data();
            size_t n_particles = comm_scheme.m_partner[p].m_sends[i].m_particle_i.size();
            for(size_t j=0;j<n_particles;j++)
            {
              assert( particle_index[j]>=0 && particle_index[j] < cells[cell_i].size() );
              cells[ cell_i ].read_tuple( particle_index[j], data->m_particles[j] );
              apply_r_shift( data->m_particles[j] , rx_shift, ry_shift, rz_shift );
            }
            data_cur += sizeof(CellParticlesUpdateData) + n_particles * sizeof(ParticleTuple);
            if( cell_scalars != nullptr )
            {
              GridCellValueType* gcv = reinterpret_cast<GridCellValueType*>( data_ptr + data_cur );
              for(unsigned int c=0;c<cell_scalar_components;c++) gcv[c]  = cell_scalars[cell_i*cell_scalar_components+c];
              data_cur += sizeof(GridCellValueType) * cell_scalar_components;
            }
            data = reinterpret_cast<CellParticlesUpdateData*>( data_ptr + data_cur );
          }
          assert(data_cur == send_buffer[p].size() );
          //ldbg << "sending "<<send_buffer_size<<" bytes to P"<<p<<std::endl;
          ++ active_sends;
          MPI_Isend( (char*) send_buffer[p].data() , send_buffer_size, MPI_CHAR, p, comm_tag, comm, & requests[nprocs+p] );
        }
      }
      
      size_t ghost_particles_recv = 0;
      while( active_sends>0 || active_recvs>0 )
      {
        int reqidx = MPI_UNDEFINED;
        MPI_Status status;
        MPI_Waitany(total_requests,requests.data(),&reqidx,&status);
        if( reqidx != MPI_UNDEFINED )
        {
          if( reqidx < nprocs ) // it's a receive
          {
            int p = reqidx;

#           ifndef NDEBUG
            size_t cells_to_receive = comm_scheme.m_partner[p].m_receives.size();
            size_t particles_to_receive = 0;
            for(size_t i=0;i<cells_to_receive;i++) { particles_to_receive += ghost_cell_receive_info(comm_scheme.m_partner[p].m_receives[i]).m_n_particles; }
            ssize_t receive_size = ( cells_to_receive * sizeof(CellParticlesUpdateData) ) + ( particles_to_receive * sizeof(ParticleTuple) );
            if( cell_scalars != nullptr )
            {
              receive_size += cells_to_receive * sizeof(GridCellValueType) * cell_scalar_components;
            }
            int status_count = 0;
            MPI_Get_count(&status,MPI_CHAR,&status_count);
            assert( receive_size == status_count );
#           endif

            uint8_t* data_ptr = receive_buffer[p].data();
            size_t data_cur = 0;
            CellParticlesUpdateData* data = reinterpret_cast<CellParticlesUpdateData*>( data_ptr + data_cur );
            for( auto cell_input_it : comm_scheme.m_partner[p].m_receives )
            {
              const auto cell_input = ghost_cell_receive_info(cell_input_it);
              assert( data_cur < receive_buffer[p].size() );
              size_t cell_i = cell_input.m_cell_i;
	      /// std::cout << " cells[cell_i].size() " << cells[cell_i].size() << " pour cell_i = " << cell_i << std::endl;
              assert( cell_i == data->m_cell_i );
              assert( cell_i>=0 && cell_i<n_cells );
              size_t n_particles = cell_input.m_n_particles;
              ghost_particles_recv += n_particles;
	      // std::cout << " n_particles " << n_particles << " avec CreateParticles " << CreateParticles << std::endl;
	      if( CreateParticles )
              {
                //assert( cells[cell_i].size() == 0 );
                cells[cell_i].resize( n_particles , grid->cell_allocator() );
              }
              else
              {
                assert( n_particles == cells[cell_i].size() );
              }
              for(size_t i=0;i<n_particles;i++)
              {
                if( CreateParticles )
                {
                  // the difference here is fields not in data->m_particles[i] are zeroed in cells[cell_i]
                  cells[cell_i].set_tuple( i, ParticleFullTuple( data->m_particles[i] ) );
                }
                else
                {
                  // while here, only fields in data->m_particles[i] are written to cells[cell_i], others are left unchanged
                  cells[cell_i].write_tuple( i, data->m_particles[i] );
                }
              }
              data_cur += sizeof(CellParticlesUpdateData) + n_particles * sizeof(ParticleTuple);
              if( cell_scalars != nullptr )
              {
                const GridCellValueType* gcv = reinterpret_cast<const GridCellValueType*>( data_ptr + data_cur );
                for(unsigned int c=0;c<cell_scalar_components;c++) cell_scalars[cell_i*cell_scalar_components+c] = gcv[c];
                data_cur += sizeof(GridCellValueType) * cell_scalar_components;
              }              
              data = reinterpret_cast<CellParticlesUpdateData*>( data_ptr + data_cur );
            }
            assert( data_cur == receive_buffer[p].size() );
            receive_buffer[p].clear();
            receive_buffer[p].shrink_to_fit();
            -- active_recvs;
            //ldbg<<"received from P"<<p<<" done, remaining recvs="<<active_recvs<< std::endl;
          }
          else // it's a send
          {
            int p = reqidx - nprocs;
            send_buffer[p].clear();
            send_buffer[p].shrink_to_fit();
            -- active_sends;
            //ldbg<<"send to P"<<p<<" done, remaining sends="<<active_sends<< std::endl;
          }
        }
      }      
      
      if( CreateParticles )
      {
        grid->rebuild_particle_offsets();
      }

      ldbg << "--- end update_ghosts : received "<< ghost_particles_recv<<" ghost particles" << std::endl;
     }

  };

}

