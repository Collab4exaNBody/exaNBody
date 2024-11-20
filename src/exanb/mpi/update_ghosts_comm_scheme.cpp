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

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/log.h>
#include <onika/math/basic_types_stream.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/particle_id_codec.h>
#include <exanb/grid_cell_particles/grid_cell_values.h>

#include <mpi.h>
#include <vector>
#include <string>
#include <list>
#include <algorithm>

#include <mpi.h>
#include <exanb/mpi/update_ghost_utils.h>
#include <exanb/mpi/ghosts_comm_scheme.h>
#include <onika/mpi/data_types.h>

namespace exanb
{
  
  using namespace UpdateGhostsUtils;

# ifndef NDEBUG
  static inline bool self_connected_block( const Domain& domain, const GridBlock& b, ssize_t gl)
  {
    return ( domain.periodic_boundary_x() && ( b.start.i + domain.grid_dimension().i - b.end.i ) < gl )
        || ( domain.periodic_boundary_y() && ( b.start.j + domain.grid_dimension().j - b.end.j ) < gl )
        || ( domain.periodic_boundary_z() && ( b.start.k + domain.grid_dimension().k - b.end.k ) < gl )
        || ( domain.mirror_x_min() && b.start.i*2 < gl )
        || ( domain.mirror_y_min() && b.start.j*2 < gl )
        || ( domain.mirror_z_min() && b.start.k*2 < gl )
        || ( domain.mirror_x_max() && ( ( domain.grid_dimension().i - b.end.i ) * 2 ) < gl )
        || ( domain.mirror_y_max() && ( ( domain.grid_dimension().j - b.end.j ) * 2 ) < gl )
        || ( domain.mirror_z_max() && ( ( domain.grid_dimension().k - b.end.k ) * 2 ) < gl );
  }
# endif

  template<typename GridT>
  struct UpdateGhostsCommSchemeNode : public OperatorNode
  {
    using ParticleTuple = typename FieldSetToParticleTuple< AddDefaultFields< FieldSet<> > >::type;
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
    ADD_SLOT( MPI_Comm , mpi                , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( long     , mpi_tag            , INPUT , 0 );
    ADD_SLOT( GridT    , grid               , INPUT , REQUIRED );
    ADD_SLOT( GridCellValues                , grid_cell_values  , INPUT , OPTIONAL );
    ADD_SLOT( Domain   , domain             , INPUT , REQUIRED );
    ADD_SLOT( bool     , enable_cell_values , INPUT , false, DocString{"if true, empty cells are included, so that per cell values can be communicated even if cell has no particles"} );
    
    // declared IN/OUT so that it is connected to the one created during init.
    // if no particle movement is triggered, the originally computed comunication scheme is used
    ADD_SLOT( GhostCommunicationScheme , ghost_comm_scheme , INPUT_OUTPUT);

    inline void execute () override final
    {
      using onika::mpi::mpi_datatype;
      
      MPI_Comm comm = *mpi;

      // start with prerequisites
      assert( check_domain(*domain) );
      assert( grid->cell_size() == domain->cell_size() );

      GridT& grid = *(this->grid);
      const Domain& domain = *(this->domain);
      GhostCommunicationScheme& comm_scheme = *ghost_comm_scheme;

      // global information needed afterward to compute buffer sizes
      comm_scheme.m_grid_dims = grid.dimension();
      comm_scheme.m_cell_bytes = sizeof( CellParticlesUpdateData );
      comm_scheme.m_particle_bytes = 0; //sizeof( ParticleTuple ); // cannot be known unitl we specify the list of attributes to update
      if( *enable_cell_values && grid_cell_values.has_value() )
      {
        comm_scheme.m_cell_bytes += grid_cell_values->components() * sizeof( typename GridCellValues::GridCellValueType );
      }

      int comm_tag = *mpi_tag;
      int nprocs = 1;
      int rank = 0;
      MPI_Comm_size(comm,&nprocs);
      MPI_Comm_rank(comm,&rank);

      bool keep_empty_cells = *enable_cell_values;

      // only own cells will be used during exchanges
      size_t ghost_layers = grid.ghost_layers();
      GridBlock my_block = enlarge_block( grid.block() , -ghost_layers );
      IJK domain_grid_dims = domain.grid_dimension();
      //Vec3d domain_size = bounds_size( domain.bounds() );

      ldbg <<"UpdateGhostsCommScheme: --- begin ghost_comm_scheme ---" << std::endl;
      ldbg <<"UpdateGhostsCommScheme: domain grid="<<domain_grid_dims<<" grid.block()="<<grid.block()<<" my_block="<<my_block << " ghost_layers=" << ghost_layers << std::endl;

      // for periodic conditions we shift simulation box once to the right or to the left
      // FIXME: this could be more than 1 and less than -1, in case the nieghborhood distance is greater than the domain itself
      assert( ! ( domain.periodic_boundary_x() && ( domain.mirror_x_min() || domain.mirror_x_max() ) ) ); // a direction cannot use periodicity and mirroring at the same time
      int periodic_i_start = -1;
      int periodic_i_end = 1;
      if( ! domain.periodic_boundary_x() && ! domain.mirror_x_min() ) { periodic_i_start = 0; }
      if( ! domain.periodic_boundary_x() && ! domain.mirror_x_max() ) { periodic_i_end   = 0; }

      assert( ! ( domain.periodic_boundary_y() && ( domain.mirror_y_min() || domain.mirror_y_max() ) ) ); // a direction cannot use periodicity and mirroring at the same time
      int periodic_j_start = -1;
      int periodic_j_end = 1;
      if( ! domain.periodic_boundary_y() && ! domain.mirror_y_min() ) { periodic_j_start = 0; }
      if( ! domain.periodic_boundary_y() && ! domain.mirror_y_max() ) { periodic_j_end   = 0; }

      assert( ! ( domain.periodic_boundary_z() && ( domain.mirror_z_min() || domain.mirror_z_max() ) ) ); // a direction cannot use periodicity and mirroring at the same time
      int periodic_k_start = -1;
      int periodic_k_end = 1;
      if( ! domain.periodic_boundary_z() && ! domain.mirror_z_min() ) { periodic_k_start = 0; }
      if( ! domain.periodic_boundary_z() && ! domain.mirror_z_max() ) { periodic_k_end   = 0; }
      
      // gather all blocks
      std::vector<GridBlock> all_blocks(nprocs);
      MPI_Allgather( (char*) &my_block, sizeof(GridBlock), MPI_CHAR, (char*) all_blocks.data(), sizeof(GridBlock), MPI_CHAR, comm);
      assert( all_blocks[rank] == my_block );

      // send buffers
      comm_scheme.m_partner.clear();
      comm_scheme.m_partner.resize( nprocs );
      std::vector< std::vector<uint64_t> > send_buffer( nprocs ); // send buffer contains an array of encoded pairs (partner cell, number of particles), for each proc ()     

      // send counts
      std::vector<size_t> send_count(nprocs);

#     pragma omp parallel //num_threads(1)
      {
        // per cell tasks will be generated in parallel by the first nprocs threads
#       pragma omp for schedule(static) //nowait
        for(int p=0;p<nprocs;p++)
        {
          size_t partner_intersecting_cells[3][3][3];
          size_t nb_intersecting_cells = 0;
          for(int k=periodic_k_start;k<=periodic_k_end;k++)
          for(int j=periodic_j_start;j<=periodic_j_end;j++)
          for(int i=periodic_i_start;i<=periodic_i_end;i++)
          {
            partner_intersecting_cells[1+k][1+j][1+i] = 0;
            if( i!=0 || j!=0 || k!=0 || p!=rank )
            {
              const IJK shift = { i,j,k };
              partner_intersecting_cells[1+k][1+j][1+i] = partner_count_intersecting_cells( all_blocks[p], grid, domain, shift );
            }
            nb_intersecting_cells += partner_intersecting_cells[1+k][1+j][1+i];
          }

          // ldbg << "P"<<p<<" : nb_intersecting_cells="<< nb_intersecting_cells <<std::endl;
          comm_scheme.m_partner[p].m_sends.resize( nb_intersecting_cells );
          send_buffer[p].resize( nb_intersecting_cells );
          size_t cur_send_item = 0;
          
          for(int k=periodic_k_start;k<=periodic_k_end;k++)
          for(int j=periodic_j_start;j<=periodic_j_end;j++)
          for(int i=periodic_i_start;i<=periodic_i_end;i++)
          {
            if( i!=0 || j!=0 || k!=0 || p!=rank )
            {
              const IJK shift = { i,j,k };
              partner_comm_scheme( all_blocks[p], grid, domain, shift, comm_scheme.m_partner[p], send_buffer[p], cur_send_item, partner_intersecting_cells[1+k][1+j][1+i] );
              cur_send_item += partner_intersecting_cells[1+k][1+j][1+i];
            }
          }
          
          if( comm_scheme.m_partner[p].m_sends.size() != nb_intersecting_cells || send_buffer[p].size() != nb_intersecting_cells || cur_send_item != nb_intersecting_cells )
          {
            fatal_error() << "Inconsistent intersecting cell count in counting step and packing step"<<std::endl;
          }
        }
        
        // from here, other threads (and first threads that finished for loop) execute generated tasks
#       pragma omp taskwait

        // all tasks done, remove empty cells if needed
        if( ! keep_empty_cells )
        {
          // ldbg << "remove empty cells" <<std::endl;
#         pragma omp for schedule(static) //nowait
          for(int p=0;p<nprocs;p++)
          {
            size_t send_buf_size = send_buffer[p].size();
            for(size_t i=0; i<send_buf_size; )
            {
              size_t cell_i = 0;
              size_t n_particles = 0;
              decode_cell_particle( send_buffer[p][i], cell_i, n_particles );
              // ldbg<<"cell #"<<cell_i<<" has "<<n_particles<<" particles"<<std::endl;
              if( n_particles == 0 )
              {
                -- send_buf_size ;
                send_buffer[p][i] = send_buffer[p][send_buf_size];
                comm_scheme.m_partner[p].m_sends[i] = std::move( comm_scheme.m_partner[p].m_sends[send_buf_size] );
              }
              else
              {
                ++i;
              }
            }
            // ldbg<<"P"<<p<<" : send_buf_size = "<<send_buffer[p].size()<<" -> "<<send_buf_size<<std::endl;
            send_buffer[p].resize( send_buf_size );
            comm_scheme.m_partner[p].m_sends.resize( send_buf_size );
          }
        }

        // fill send_count array
#       pragma omp for schedule(static) //nowait
        for(int p=0;p<nprocs;p++)
        {
          send_count[p] = send_buffer[p].size();
          // ldbg << "P"<<p<<" : send_count="<<send_count[p]<<std::endl;
        }

      } // end of parallel section

      // store send buffer sizes in an array for alltoall operation

      // what to check here ?
      // 1. a source cell cell_i, for a partner p, is either not used or used exactly once.
      // 2. a source cell cell_i appears at most 7 times for all exchanges

      // build receive counts from send counts
      std::vector<size_t> recv_count(nprocs,0);
      assert( send_count[rank] == 0 || self_connected_block(domain,my_block,ghost_layers) );
      MPI_Alltoall( send_count.data(), 1, mpi_datatype<size_t>(), recv_count.data(), 1, mpi_datatype<size_t>(), comm );
      assert( recv_count[rank] == 0 || self_connected_block(domain,my_block,ghost_layers) );

      // initialize MPI requests for both sends and receives
      size_t total_requests = 2 * nprocs;
      std::vector< MPI_Request > requests( total_requests , MPI_REQUEST_NULL );
      total_requests = 0;

      // alocate receive buffers and start async receives and sends
      size_t active_recvs=0, active_sends=0;
      for(int p=0;p<nprocs;p++)
      {
        comm_scheme.m_partner[p].m_receives.resize( recv_count[p] );
        if( p != rank )
        {
          if( recv_count[p] > 0 )
          {
            //receive_buffer[p].resize( recv_count[p] );
            MPI_Irecv( comm_scheme.m_partner[p].m_receives.data() /*receive_buffer[p].data()*/ , recv_count[p], mpi_datatype<uint64_t>(), p, comm_tag, comm, & requests[total_requests++] );
            ++ active_recvs;
          }
          if( send_count[p] > 0 )
          {
            MPI_Isend( send_buffer[p].data() , send_count[p], mpi_datatype<uint64_t>(), p, comm_tag, comm, & requests[total_requests++] );
            ++ active_sends;
          }
        }
        else // buffer to send to myself ( p == rank )
        {
          if( send_count[p] != recv_count[p] )
          {
            fatal_error()<<"inconsistent send/receive sizes for self message send."<<std::endl;
          }
          if( recv_count[p] > 0 )
          {
            for(size_t i=0;i<recv_count[p];i++) comm_scheme.m_partner[p].m_receives[i] = send_buffer[p][i];
          }
          send_buffer[p].clear();
        }
      }
      
      
      // wait for asynchronous operations to finish
      /********************************************************************/
      ldbg << "active_sends="<<active_sends<<" , active_recvs="<<active_recvs<<" , total_requests="<<total_requests<<std::endl;
      std::vector<MPI_Status> request_status_array( total_requests );
      MPI_Waitall( total_requests , requests.data() , request_status_array.data() );
      for(auto & sbuf : send_buffer) sbuf.clear();
      active_sends = 0;
      active_recvs = 0;
      /********************************************************************/


      // final step, rebuild buffer particle offsets
      /********************************************************************/
      for(int p=0;p<nprocs;p++)
      {
        const size_t cells_to_send = comm_scheme.m_partner[p].m_sends.size();
        size_t total_particles = 0;
        for(size_t i=0;i<cells_to_send;i++)
        {
          comm_scheme.m_partner[p].m_sends[i].m_send_buffer_offset = total_particles; 
          total_particles += comm_scheme.m_partner[p].m_sends[i].m_particle_i.size();; 
        }
        const size_t cells_to_receive = comm_scheme.m_partner[p].m_receives.size();
        comm_scheme.m_partner[p].m_receive_offset.assign( cells_to_receive , 0 );
        total_particles = 0;
        for(size_t i=0;i<cells_to_receive;i++)
        {
          const auto cell_input_it = comm_scheme.m_partner[p].m_receives[i];
          const auto cell_input = ghost_cell_receive_info(cell_input_it);
          comm_scheme.m_partner[p].m_receive_offset[i] = total_particles;
          total_particles += cell_input.m_n_particles;
        }
      }
      /********************************************************************/

      ldbg << "UpdateGhostsCommScheme: --- end ghost_comm_scheme ---" << std::endl;
    }



    // ----------------------------------------------------        
    inline size_t partner_count_intersecting_cells(
      const GridBlock& partner_block,
      const GridT& grid,
      const Domain& domain,
      const IJK& dom_shift )
    {
      // build portion of local sub domain intersecting one of the ghost replica of computation domain
      assert( dom_shift.i>=-1 && dom_shift.i<=1 && dom_shift.j>=-1 && dom_shift.j<=1 && dom_shift.k>=-1 && dom_shift.k<=1 );
      size_t n_intersecting_cells = 0;

      const auto partner_block_with_ghosts = enlarge_block( partner_block , grid.ghost_layers() );      
      GRID_FOR_BEGIN( grid.dimension()-2*grid.ghost_layers() , _ , loc )
      {
        const auto my_cell_loc = loc + grid.ghost_layers();
        const IJK domain_cell_loc = my_cell_loc + grid.offset();
        IJK shifted_cell_loc = ( dom_shift * domain.grid_dimension() ) + domain_cell_loc;
        if( dom_shift.i == -1 && domain.mirror_x_min() ) shifted_cell_loc.i =                               - domain_cell_loc.i - 1;
        if( dom_shift.i ==  1 && domain.mirror_x_max() ) shifted_cell_loc.i = 2 * domain.grid_dimension().i - domain_cell_loc.i - 1;
        if( dom_shift.j == -1 && domain.mirror_y_min() ) shifted_cell_loc.j =                               - domain_cell_loc.j - 1;
        if( dom_shift.j ==  1 && domain.mirror_y_max() ) shifted_cell_loc.j = 2 * domain.grid_dimension().j - domain_cell_loc.j - 1;
        if( dom_shift.k == -1 && domain.mirror_z_min() ) shifted_cell_loc.k =                               - domain_cell_loc.k - 1;
        if( dom_shift.k ==  1 && domain.mirror_z_max() ) shifted_cell_loc.k = 2 * domain.grid_dimension().k - domain_cell_loc.k - 1;
                
        if( inside_block( partner_block_with_ghosts , shifted_cell_loc ) )
        {
          // partner cell must be identified as a ghost cell according to its own grid (thus must not be an inner cell)
          assert( ! inside_block(partner_block,shifted_cell_loc) );
          // a ghost cell in a partner's ghost area must correspond to an inner cell of mine
          assert( ! grid.is_ghost_cell(my_cell_loc) );
          // account for intersected cell
          ++ n_intersecting_cells;
        }
      }
      GRID_FOR_END

      return n_intersecting_cells;
    }

    // ----------------------------------------------------        
    inline void partner_comm_scheme(
      const GridBlock& partner_block,
      const GridT& grid,
      const Domain& domain,
      const IJK& dom_shift,
      GhostPartnerCommunicationScheme& comm_scheme,
      std::vector<uint64_t>& send_buffer,
      size_t cur_send_item,
      size_t expected_partner_nb_cells )
    {
      // build portion of local sub domain intersecting one of the ghost replica of computation domain
      assert( dom_shift.i>=-1 && dom_shift.i<=1 && dom_shift.j>=-1 && dom_shift.j<=1 && dom_shift.k>=-1 && dom_shift.k<=1 );

      auto * pcomm_scheme = & comm_scheme;
      auto * psend_buffer = & send_buffer;
      const auto * pgrid = & grid;
      const auto * pdomain = & domain;
#     pragma omp task default(none) firstprivate(partner_block,dom_shift,cur_send_item,expected_partner_nb_cells /*shared*/,pcomm_scheme,psend_buffer,pgrid,pdomain) 
      {
        auto & comm_scheme = *pcomm_scheme;
        auto & send_buffer = *psend_buffer;
        const auto & grid = *pgrid;
        const auto & domain = *pdomain;
        
        const GhostBoundaryModifier boundary = { domain.origin() , domain.extent() };
        
        const AABB partner_inner_bounds = block_to_bounds( partner_block , grid.origin() , grid.cell_size() );
        const AABB partner_outter_bounds = enlarge( partner_inner_bounds , grid.max_neighbor_distance() );

        uint32_t cell_boundary_flags = 0;
        if( dom_shift.i == -1 && domain.periodic_boundary_x() ) cell_boundary_flags |= GhostBoundaryModifier:: SHIFT_X ;
        if( dom_shift.i ==  1 && domain.periodic_boundary_x() ) cell_boundary_flags |= GhostBoundaryModifier:: SHIFT_X | GhostBoundaryModifier::SIDE_X;
        if( dom_shift.i == -1 && domain.mirror_x_min()        ) cell_boundary_flags |= GhostBoundaryModifier::MIRROR_X ;
        if( dom_shift.i ==  1 && domain.mirror_x_max()        ) cell_boundary_flags |= GhostBoundaryModifier::MIRROR_X | GhostBoundaryModifier::SIDE_X;

        if( dom_shift.j == -1 && domain.periodic_boundary_y() ) cell_boundary_flags |= GhostBoundaryModifier:: SHIFT_Y ;
        if( dom_shift.j ==  1 && domain.periodic_boundary_y() ) cell_boundary_flags |= GhostBoundaryModifier:: SHIFT_Y | GhostBoundaryModifier::SIDE_Y;
        if( dom_shift.j == -1 && domain.mirror_y_min()        ) cell_boundary_flags |= GhostBoundaryModifier::MIRROR_Y ;
        if( dom_shift.j ==  1 && domain.mirror_y_max()        ) cell_boundary_flags |= GhostBoundaryModifier::MIRROR_Y | GhostBoundaryModifier::SIDE_Y;

        if( dom_shift.k == -1 && domain.periodic_boundary_z() ) cell_boundary_flags |= GhostBoundaryModifier:: SHIFT_Z ;
        if( dom_shift.k ==  1 && domain.periodic_boundary_z() ) cell_boundary_flags |= GhostBoundaryModifier:: SHIFT_Z | GhostBoundaryModifier::SIDE_Z;
        if( dom_shift.k == -1 && domain.mirror_z_min()        ) cell_boundary_flags |= GhostBoundaryModifier::MIRROR_Z ;
        if( dom_shift.k ==  1 && domain.mirror_z_max()        ) cell_boundary_flags |= GhostBoundaryModifier::MIRROR_Z | GhostBoundaryModifier::SIDE_Z;
        
        const auto partner_block_with_ghosts = enlarge_block( partner_block , grid.ghost_layers() );      
        const auto* cells = grid.cells();
        
        size_t n_intersecting_cells = 0;

        GRID_FOR_BEGIN( grid.dimension()-2*grid.ghost_layers() , _ , loc )
        {
          const auto my_cell_loc = loc + grid.ghost_layers();
          assert( grid.contains(my_cell_loc) );
          const IJK domain_cell_loc = my_cell_loc + grid.offset();
          IJK shifted_cell_loc = ( dom_shift * domain.grid_dimension() ) + domain_cell_loc;
          if( dom_shift.i == -1 && domain.mirror_x_min() ) shifted_cell_loc.i =                               - domain_cell_loc.i - 1;
          if( dom_shift.i ==  1 && domain.mirror_x_max() ) shifted_cell_loc.i = 2 * domain.grid_dimension().i - domain_cell_loc.i - 1;
          if( dom_shift.j == -1 && domain.mirror_y_min() ) shifted_cell_loc.j =                               - domain_cell_loc.j - 1;
          if( dom_shift.j ==  1 && domain.mirror_y_max() ) shifted_cell_loc.j = 2 * domain.grid_dimension().j - domain_cell_loc.j - 1;
          if( dom_shift.k == -1 && domain.mirror_z_min() ) shifted_cell_loc.k =                               - domain_cell_loc.k - 1;
          if( dom_shift.k ==  1 && domain.mirror_z_max() ) shifted_cell_loc.k = 2 * domain.grid_dimension().k - domain_cell_loc.k - 1;

          if( inside_block( partner_block_with_ghosts , shifted_cell_loc ) )
          {
            // partner cell must be identified as a ghost cell according to its own grid (thus not an inner cell)
            assert( ! inside_block(partner_block,shifted_cell_loc) );
            // a ghost cell in a partner's ghost area must correspond to an inner cell of mine
            assert( ! grid.is_ghost_cell(my_cell_loc) );
            // account for intersected cell

            const ssize_t my_cell_i = grid_ijk_to_index( grid.dimension() , my_cell_loc );
            const ssize_t partner_cell_i = grid_ijk_to_index( dimension(partner_block_with_ghosts) , shifted_cell_loc - partner_block_with_ghosts.start );
            const size_t n_particles = cells[my_cell_i].size();
            const auto* __restrict__ rx = cells[my_cell_i][field::rx];
            const auto* __restrict__ ry = cells[my_cell_i][field::ry];
            const auto* __restrict__ rz = cells[my_cell_i][field::rz];

            GhostCellSendScheme send_scheme; 
            send_scheme.m_cell_i = my_cell_i;
            send_scheme.m_partner_cell_i = partner_cell_i;
            send_scheme.m_flags = cell_boundary_flags;

            for(size_t p_i=0;p_i<n_particles;p_i++)
            {
              const Vec3d r{ rx[p_i], ry[p_i], rz[p_i] };
              const Vec3d ghost_r = boundary.apply_r_modifier(r,cell_boundary_flags);
              if( is_inside( partner_outter_bounds, ghost_r ) )
              {
                send_scheme.m_particle_i.push_back( p_i );
              }
            }
            size_t n_particles_to_send = send_scheme.m_particle_i.size();
            assert( cur_send_item < comm_scheme.m_sends.size() );
            assert( cur_send_item < send_buffer.size() );
            comm_scheme.m_sends[ cur_send_item ] = std::move(send_scheme);
            send_buffer[ cur_send_item ] = encode_cell_particle( partner_cell_i , n_particles_to_send );
            // ldbg_stream() << "send_buffer["<<cur_send_item<<"] = "<<partner_cell_i<<","<<n_particles_to_send /*send_buffer[ cur_send_item ]*/ <<std::endl;
            ++ n_intersecting_cells;
            ++ cur_send_item;
          }
        }
        GRID_FOR_END
        
        if( n_intersecting_cells != expected_partner_nb_cells )
        {
          fatal_error() << "expected "<<expected_partner_nb_cells<<" intersecing cells with ghost partner, but found "<<n_intersecting_cells<<" in pack task"<<std::endl;
        }
      } // end of omp task
      
    }
    // ------------------------

  };

  // === register factory ===
  ONIKA_AUTORUN_INIT(update_ghosts_comm_scheme)
  {
    OperatorNodeFactory::instance()->register_factory(
      "ghost_comm_scheme",
      make_grid_variant_operator< UpdateGhostsCommSchemeNode > );
  }

}

