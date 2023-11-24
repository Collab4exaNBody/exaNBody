#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/log.h>
#include <exanb/core/basic_types_stream.h>
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
#include <exanb/mpi/data_types.h>

namespace exanb
{
  
  using namespace UpdateGhostsUtils;

# ifndef NDEBUG
  // FIXME: not sure about this. shouldn't ghost_layers be part of equation ?
  static inline bool self_connected_block( const Domain& domain, const GridBlock& b)
  {
    return ( domain.periodic_boundary_x() && b.start.i==0 && b.end.i==domain.grid_dimension().i )
        || ( domain.periodic_boundary_y() && b.start.j==0 && b.end.j==domain.grid_dimension().j )
        || ( domain.periodic_boundary_z() && b.start.k==0 && b.end.k==domain.grid_dimension().k );
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
      Vec3d domain_size = bounds_size( domain.bounds() );

      ldbg <<"UpdateGhostsCommScheme: --- begin ghost_comm_scheme ---" << std::endl;
      ldbg <<"UpdateGhostsCommScheme: domain grid="<<domain_grid_dims<<" grid.block()="<<grid.block()<<" my_block="<<my_block << "ghost_layers=" << ghost_layers << std::endl;

      // for periodic conditions we shift simulation box once to the right or to the left
      // FIXME: this could be more than 1 and less than -1, in case the nieghborhood distance is greater than the domain itself
      int periodic_i_start = -1;
      int periodic_i_end = 1;
      int periodic_j_start = -1;
      int periodic_j_end = 1;
      int periodic_k_start = -1;
      int periodic_k_end = 1;
      if( ! domain.periodic_boundary_x() ) { periodic_i_start = periodic_i_end = 0;  }
      if( ! domain.periodic_boundary_y() ) { periodic_j_start = periodic_j_end = 0;  }
      if( ! domain.periodic_boundary_z() ) { periodic_k_start = periodic_k_end = 0;  }

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
          size_t nb_intersecting_cells = 0;
          for(int k=periodic_k_start;k<=periodic_k_end;k++)
          for(int j=periodic_j_start;j<=periodic_j_end;j++)
          for(int i=periodic_i_start;i<=periodic_i_end;i++)
          {
            IJK shift { i,j,k };
            if( i!=0 || j!=0 || k!=0 || p!=rank )
            {
              Vec3d r_shift = domain_size * shift;
              IJK block_shift = shift * domain_grid_dims;
              size_t partner_intersecting_cells = partner_count_intersecting_cells( all_blocks[p], block_shift, r_shift, grid );
              /*if(partner_intersecting_cells>0)
              {
                ldbg<<"rank="<<rank<<" p="<<p<<" block_shift="<<block_shift<<" r_shift="<<r_shift<<" inter cells="<<partner_intersecting_cells<<"\n";
              }*/
              nb_intersecting_cells += partner_intersecting_cells;
            }
          }

          // ldbg << "P"<<p<<" : nb_intersecting_cells="<< nb_intersecting_cells <<std::endl;
          comm_scheme.m_partner[p].m_sends.resize( nb_intersecting_cells );
          send_buffer[p].resize( nb_intersecting_cells );
          size_t cur_send_item = 0;
          
          for(int k=periodic_k_start;k<=periodic_k_end;k++)
          for(int j=periodic_j_start;j<=periodic_j_end;j++)
          for(int i=periodic_i_start;i<=periodic_i_end;i++)
          {
            IJK shift { i,j,k };
            if( i!=0 || j!=0 || k!=0 || p!=rank )
            {
              Vec3d r_shift = domain_size * shift;
              IJK block_shift = shift * domain_grid_dims;
              cur_send_item += partner_comm_scheme( all_blocks[p], block_shift, r_shift, grid, comm_scheme.m_partner[p], send_buffer[p], cur_send_item );
            }
          }
          
          if( comm_scheme.m_partner[p].m_sends.size() != nb_intersecting_cells || send_buffer[p].size() != nb_intersecting_cells || cur_send_item != nb_intersecting_cells )
          {
            lerr << "Internal error: bad intersecting cell count"<<std::endl;
            std::abort();
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
      assert( send_count[rank] == 0 || self_connected_block(domain,my_block) );
      MPI_Alltoall( send_count.data(), 1, mpi_datatype<size_t>(), recv_count.data(), 1, mpi_datatype<size_t>(), comm );
      assert( recv_count[rank] == 0 || self_connected_block(domain,my_block) );

      // initialize MPI requests for both sends and receives
      size_t total_requests = 2 * nprocs;
      std::vector< MPI_Request > requests( total_requests , MPI_REQUEST_NULL );
      total_requests = 0;

      // alocate receive buffers and start async receives and sends
      size_t active_recvs=0, active_sends=0;
      for(int p=0;p<nprocs;p++)
      {
        if( p == rank )
        {
          if( recv_count[p] != send_count[p] )
          {
            fatal_error() << "Inconsistent self send/receive sizes : recv_count["<<p<<"]="<<recv_count[p]<<" , send_count["<<p<<"]="<<send_count[p]<<std::endl;
          }
          if( recv_count[p] > 0 )
          {
            comm_scheme.m_partner[p].m_receives.resize( recv_count[p] );
            for(size_t i=0;i<recv_count[p]; i++) comm_scheme.m_partner[p].m_receives[i] = send_buffer[p][i];
            send_buffer[p].clear();
          }
        }
        else
        {      
          comm_scheme.m_partner[p].m_receives.resize( recv_count[p] );
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
      }
      
      ldbg << "UpdateGhostsCommScheme: active_sends="<<active_sends<<" active_recvs="<<active_recvs<<std::endl;

      // simpler version. possible because we do not decode recevied buffers anymore. they are received in their final form.
      std::vector<MPI_Status> req_status( total_requests );
      MPI_Waitall( total_requests , requests.data() , req_status.data() );
      for(auto& sbuf:send_buffer) sbuf.clear();

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
      const IJK& block_shift,
      const Vec3d& r_shift,
      const GridT& grid )
    {
      ssize_t ghost_layers = grid.ghost_layers();
      GridBlock my_block = enlarge_block( grid.block() , -ghost_layers );
      GridBlock shifted_my_block = my_block + block_shift;
      GridBlock partner_block_with_ghosts = enlarge_block( partner_block , ghost_layers );
//      IJK partner_block_with_ghosts_dims = dimension( partner_block_with_ghosts );
      GridBlock intersect_block_ghosts = intersection( partner_block_with_ghosts , shifted_my_block );
      ssize_t n_intersecting_cells = 0;
      if( ! is_empty(intersect_block_ghosts) )
      {
        IJK intersect_block_ghosts_dims = dimension(intersect_block_ghosts);
        n_intersecting_cells = grid_cell_count( intersect_block_ghosts_dims );
        assert( n_intersecting_cells>=0 );
      }      
      return n_intersecting_cells;
    }

    // ----------------------------------------------------        
    inline size_t partner_comm_scheme(
      const GridBlock& partner_block,
      const IJK& block_shift,
      const Vec3d& r_shift,
      const GridT& grid,
      GhostPartnerCommunicationScheme& comm_scheme,
      std::vector<uint64_t>& send_buffer,
      size_t cur_send_item
      )
    {
      ssize_t ghost_layers = grid.ghost_layers();
      GridBlock my_block = enlarge_block( grid.block() , -ghost_layers );
      GridBlock shifted_my_block = my_block + block_shift;
      GridBlock partner_block_with_ghosts = enlarge_block( partner_block , ghost_layers );
      GridBlock intersect_block_ghosts = intersection( partner_block_with_ghosts , shifted_my_block );
      size_t n_intersecting_cells = 0;
 
      if( ! is_empty(intersect_block_ghosts) )
      {
        IJK intersect_block_ghosts_dims = dimension(intersect_block_ghosts);
        n_intersecting_cells = grid_cell_count( intersect_block_ghosts_dims );

        // Note: it seems that it is a bad idea to use shared clause with references when using Intel compiler 20.0.0 (19.xxx based)
//#       pragma omp task default(none) firstprivate(r_shift,partner_block,block_shift,cur_send_item) shared(comm_scheme,send_buffer,grid) // if(0)
        auto * pcomm_scheme = & comm_scheme;
        auto * psend_buffer = & send_buffer;
        const auto * pgrid = & grid;
#       pragma omp task default(none) firstprivate(r_shift,partner_block,block_shift,cur_send_item /*shared*/,pcomm_scheme,psend_buffer,pgrid) 
        {
          auto & comm_scheme = *pcomm_scheme;
          auto & send_buffer = *psend_buffer;
          const auto & grid = *pgrid;
          
          const auto* cells = grid.cells();
          IJK my_dims = grid.dimension();
          ssize_t ghost_layers = grid.ghost_layers();

          Vec3d origin = grid.origin();
          double cell_size = grid.cell_size();
          IJK grid_offset = grid.offset();
          double max_nbh_dist = grid.max_neighbor_distance();

          GridBlock my_block = enlarge_block( grid.block() , -ghost_layers );
          GridBlock shifted_my_block = my_block + block_shift;
          GridBlock partner_block_with_ghosts = enlarge_block( partner_block , ghost_layers );
          GridBlock intersect_block_ghosts = intersection( partner_block_with_ghosts , shifted_my_block );
          IJK partner_block_with_ghosts_dims = dimension( partner_block_with_ghosts );

          AABB partner_inner_bounds = block_to_bounds( partner_block , origin, cell_size );
          AABB partner_outter_bounds = enlarge( partner_inner_bounds , max_nbh_dist );

          IJK intersect_block_ghosts_dims = dimension(intersect_block_ghosts);

#         ifndef NDEBUG
          AABB partner_inner_bounds_minus_threshold = enlarge( partner_inner_bounds , - grid.epsilon_cell_size() );
#         endif

          GRID_FOR_BEGIN(intersect_block_ghosts_dims,_,loc)
          {
            IJK partner_grid_loc = loc + intersect_block_ghosts.start - partner_block_with_ghosts.start;
            ssize_t partner_cell_i = grid_ijk_to_index( partner_block_with_ghosts_dims , partner_grid_loc );
            // only ghost cells can overlap with non ghost cells from another domain
            assert( inside_grid_shell( partner_block_with_ghosts_dims,0, ghost_layers, partner_grid_loc ) );

            IJK grid_loc = loc + intersect_block_ghosts.start - grid_offset - block_shift;
            ssize_t cell_i = grid_ijk_to_index( my_dims, grid_loc );
            assert( grid.contains(grid_loc) );
            size_t n_particles = cells[cell_i].size();

            const auto* __restrict__ rx = cells[cell_i][field::rx];
            const auto* __restrict__ ry = cells[cell_i][field::ry];
            const auto* __restrict__ rz = cells[cell_i][field::rz];

            // we use std::move instead of removing element after adding it
            GhostCellSendScheme send_scheme; // = comm_scheme.m_sends.back();
            send_scheme.m_cell_i = cell_i;
            send_scheme.m_partner_cell_i = partner_cell_i;
            send_scheme.m_x_shift = r_shift.x;
            send_scheme.m_y_shift = r_shift.y;
            send_scheme.m_z_shift = r_shift.z;

            for(size_t p_i=0;p_i<n_particles;p_i++)
            {
              Vec3d r{ rx[p_i], ry[p_i], rz[p_i] };
              r = r + r_shift;
              if( is_inside( partner_outter_bounds , r ) )
              {
                assert( ! is_inside_exclude_upper( partner_inner_bounds_minus_threshold , r ) );
                send_scheme.m_particle_i.push_back( p_i );
              }
            }
            size_t n_particles_to_send = send_scheme.m_particle_i.size();
            assert( cur_send_item < comm_scheme.m_sends.size() );
            assert( cur_send_item < send_buffer.size() );
            comm_scheme.m_sends[ cur_send_item ] = std::move(send_scheme);
            send_buffer[ cur_send_item ] = encode_cell_particle( partner_cell_i , n_particles_to_send );
            // ldbg_stream() << "send_buffer["<<cur_send_item<<"] = "<<partner_cell_i<<","<<n_particles_to_send /*send_buffer[ cur_send_item ]*/ <<std::endl;
            ++ cur_send_item;
          }
          GRID_FOR_END
        } // end of omp task
        
      } // ! is_empty( intersect_block_ghosts )
      
      return n_intersecting_cells;
    }
    // ------------------------

  };

  // === register factory ===
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory(
      "ghost_comm_scheme",
      make_grid_variant_operator< UpdateGhostsCommSchemeNode > );
  }

}

