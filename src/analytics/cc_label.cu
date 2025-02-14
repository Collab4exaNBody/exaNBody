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
#include <exanb/core/domain.h>
#include <exanb/grid_cell_particles/grid_cell_values.h>
#include <exanb/core/grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/mpi/grid_update_ghosts.h>
#include <exanb/analytics/cc_info.h>
#include <exanb/grid_cell_particles/cell_particle_update_functor.h>
#include <exanb/mpi/grid_update_from_ghosts.h>

#include <mpi.h>

namespace exanb
{
  using onika::scg::OperatorNode;
  using onika::scg::OperatorNodeFactory;
  using onika::scg::make_simple_operator;

  class ConnectedComponentLabel : public OperatorNode
  {
    using UpdateGhostsScratch = typename UpdateGhostsUtils::UpdateGhostsScratch;

    ADD_SLOT( MPI_Comm       , mpi                 , INPUT , MPI_COMM_WORLD , DocString{"MPI communicator"} );
    ADD_SLOT( Domain         , domain              , INPUT , REQUIRED );
    ADD_SLOT( std::string    , grid_cell_field     , INPUT , "density" , DocString{"grid cell value field to act as coonnected component mask"} );
    ADD_SLOT( double         , grid_cell_threshold , INPUT , 1. , DocString{"Treshold to determine wheter a cell is selected or not as part of a connected component"} );
    ADD_SLOT( GridCellValues , grid_cell_values    , INPUT_OUTPUT );

    ADD_SLOT( ConnectedComponentTable , cc_table   , INPUT_OUTPUT );
    ADD_SLOT( long                    , cc_count_threshold  , INPUT , 1 ); // CC that have less or equal cells than this value, are ignored

    ADD_SLOT( GhostCommunicationScheme , ghost_comm_scheme , INPUT , REQUIRED );
    ADD_SLOT( UpdateGhostsScratch      , ghost_comm_buffers, PRIVATE );
    
  public:

    // -----------------------------------------------
    // -----------------------------------------------
    inline void execute ()  override final
    {
      // check that specified field exists
      if( ! grid_cell_values->has_field( *grid_cell_field ) )
      {
        fatal_error() << "No field named '"<< *grid_cell_field << "' found in grid_cell_values"<<std::endl;
      }
      
      // we cannot handle mirroring by now
      if( domain->mirror_x_min() || domain->mirror_x_max()
       || domain->mirror_y_min() || domain->mirror_y_max()
       || domain->mirror_z_min() || domain->mirror_z_max() )
      {
        fatal_error() << "domain mirror boundary conditions are not supported yet for connected component analysis"<<std::endl;
      }

/*
      // DEBUG ONLY
      // test if empty cells are also "ghost updated"
      size_t n_empty_send_cells = 0;
      for(const auto & partner : ghost_comm_scheme->m_partner)
      {
        for(const auto send_info : partner.m_sends)
        {
          if( send_info.m_particle_i.empty() ) ++ n_empty_send_cells;
        }
      }
      ldbg << "n_empty_send_cells=" << n_empty_send_cells << std::endl;
*/
      
      // cell size (alle cells are cubical)
      const double cell_size = domain->cell_size(); 
      
      // local processor's grid dimensions, including ghost cells
      const IJK grid_dims = grid_cell_values->grid_dims(); 
      
      // local processor's grid position in simulation's grid
      const IJK grid_offset = grid_cell_values->grid_offset();
      
      // simulation's grid dimensions
      const IJK domain_dims = domain->grid_dimension(); 
      
       // ghost layers
      const ssize_t gl = grid_cell_values->ghost_layers();
      
      // number of subdivisions, in each directions, applied on cells
      const ssize_t subdiv = grid_cell_values->field( *grid_cell_field ).m_subdiv;
      
      // side size of a sub-cell
      const double subcell_size = cell_size / subdiv;
      
      // sub-cell volume
      // const double subcell_volume = subcell_size * subcell_size * subcell_size;
      
      // dimension of the subdivided simulation's grid
      const IJK domain_subdiv_dims = domain_dims * subdiv;

      // some debug information
      ldbg << "cc_label: gl="<<gl<<", cell_size="<<cell_size<<", subdiv="<<subdiv<<", subcell_size="
           <<subcell_size<<", grid_dims="<<grid_dims<<", grid_offset="<<grid_offset<<", domain_dims="<<domain_dims<<", domain_subdiv_dims="<<domain_subdiv_dims<<std::endl;

      // create additional data field for connected component label
      if( ! grid_cell_values->has_field("cc_label") )
      {
        grid_cell_values->add_field("cc_label",subdiv,1);
      }
      
      // cc_label field data accessor.
      const auto cc_label_accessor = grid_cell_values->field_data("cc_label");
      double * __restrict__ cc_label_ptr = cc_label_accessor.m_data_ptr;
      [[maybe_unused]] const size_t cc_label_stride = cc_label_accessor.m_stride;

      // density field data accessor.
      const auto density_accessor = grid_cell_values->field_data( *grid_cell_field );
      double  * __restrict__ density_ptr = density_accessor.m_data_ptr;
      const size_t density_stride = density_accessor.m_stride;
      
      // sanity check
      assert( cc_label_stride == density_stride );
      const size_t stride = density_stride; // for simplicity

      // get communicator size and local rank
      int rank=0, nprocs=1;
      MPI_Comm_rank( *mpi , &rank );
      MPI_Comm_size( *mpi , &nprocs );
      assert( rank < nprocs );

      // null grid and functors needed for ghost updates
      auto pecfunc = [self=this](auto ... args) { return self->parallel_execution_context(args ...); };
      auto pesfunc = [self=this](unsigned int i) { return self->parallel_execution_stream(i); };
      Grid< FieldSet<> > * null_grid_ptr = nullptr;
      onika::FlatTuple<> update_fields = {};

      // does the projected cell value has values in the ghost area ?
      // create a unique label id for each cell satisfying selection criteria
      unsigned long long nonzero_ghost_subcells = 0;
#     pragma omp parallel for collapse(3) schedule(static) reduction(+:nonzero_ghost_subcells)
      for( ssize_t k=0 ; k < grid_dims.k ; k++)
      for( ssize_t j=0 ; j < grid_dims.j ; j++)
      for( ssize_t i=0 ; i < grid_dims.i ; i++)
      {
        const bool is_ghost = (i<gl) || (i>=(grid_dims.i-gl)) || (j<gl) || (j>=(grid_dims.j-gl)) || (k<gl) || (k>=(grid_dims.k-gl));
        if( is_ghost )
        {
          const ssize_t grid_cell_index = grid_ijk_to_index( grid_dims , IJK{i,j,k} );
          for( ssize_t sk=0 ; sk<subdiv ; sk++)
          for( ssize_t sj=0 ; sj<subdiv ; sj++)
          for( ssize_t si=0 ; si<subdiv ; si++)
          {
            const ssize_t subcell_index = grid_ijk_to_index( IJK{subdiv,subdiv,subdiv} , IJK{si,sj,sk} );
            const ssize_t value_index = grid_cell_index * stride + subcell_index;
            if( density_ptr[value_index] > 0.0 ) ++ nonzero_ghost_subcells;
          }
        }
      }
      MPI_Allreduce(MPI_IN_PLACE,&nonzero_ghost_subcells,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,*mpi);
      
      // if needed, propagate ghost cell scalars to inner cells using additive update.
      if( nonzero_ghost_subcells > 0 )
      {
        ldbg << "nonzero_ghost_subcells="<<nonzero_ghost_subcells<<", grid_update_from_ghosts with UpdateValueAdd merge functor"<<std::endl;
        grid_update_from_ghosts( ::exanb::ldbg, *mpi, *ghost_comm_scheme, null_grid_ptr, *domain, grid_cell_values.get_pointer(),
                          *ghost_comm_buffers, pecfunc,pesfunc, update_fields,
                          0, false, false, false,
                          true, false, UpdateValueAdd{} );
#       pragma omp parallel for collapse(3) schedule(static) reduction(+:nonzero_ghost_subcells)
        for( ssize_t k=0 ; k < grid_dims.k ; k++)
        for( ssize_t j=0 ; j < grid_dims.j ; j++)
        for( ssize_t i=0 ; i < grid_dims.i ; i++)
        {
          const bool is_ghost = (i<gl) || (i>=(grid_dims.i-gl)) || (j<gl) || (j>=(grid_dims.j-gl)) || (k<gl) || (k>=(grid_dims.k-gl));
          if( is_ghost )
          {
            const ssize_t grid_cell_index = grid_ijk_to_index( grid_dims , IJK{i,j,k} );
            for( ssize_t sk=0 ; sk<subdiv ; sk++)
            for( ssize_t sj=0 ; sj<subdiv ; sj++)
            for( ssize_t si=0 ; si<subdiv ; si++)
            {
              const ssize_t subcell_index = grid_ijk_to_index( IJK{subdiv,subdiv,subdiv} , IJK{si,sj,sk} );
              const ssize_t value_index = grid_cell_index * stride + subcell_index;
              density_ptr[value_index] = 0.0;
            }
          }
        }
      }

      // density threshold to select cells
      const double threshold = *grid_cell_threshold ;
      // other usefull values
      const size_t subdiv3 = subdiv * subdiv * subdiv;
      static constexpr double ghost_no_label = -2.0;
      static constexpr double no_label = -1.0;
      
      // these 3 funcions define how we build unique ids such a way that :
      // 1. owner rank can be guessed from the unique_id
      // 2. final (stripped) unique ids will be deterministic and independant from parallel settings
      auto encode_unique_id = [&](uint64_t domain_cell_index, uint64_t grid_cell_index, uint64_t subcell_index) -> uint64_t
      {
        assert(rank>=0 && rank<nprocs);
        return ( domain_cell_index * subdiv3 + subcell_index ) * nprocs + rank;
      };
      // now we can build a function to determine final destination (process rank) for each CC based on its label
      auto owner_from_unique_id = [&](uint64_t unique_id) -> int 
      {
        return unique_id % nprocs;
      };      
      auto strip_unique_id = [&](uint64_t unique_id) -> uint64_t 
      {
        return unique_id / nprocs;
      };

      // create a unique label id for each cell satisfying selection criteria
#     pragma omp parallel for collapse(3) schedule(static)
      for( ssize_t k=0 ; k < grid_dims.k ; k++)
      for( ssize_t j=0 ; j < grid_dims.j ; j++)
      for( ssize_t i=0 ; i < grid_dims.i ; i++)
      {
        const bool is_ghost = (i<gl) || (i>=(grid_dims.i-gl)) || (j<gl) || (j>=(grid_dims.j-gl)) || (k<gl) || (k>=(grid_dims.k-gl));
        const ssize_t grid_cell_index = is_ghost ? 0 : grid_ijk_to_index( grid_dims , IJK{i,j,k} );
        const ssize_t domain_cell_index = is_ghost ? 0 : grid_ijk_to_index( domain_dims , IJK{i,j,k} - grid_offset );
        for( ssize_t sk=0 ; sk<subdiv ; sk++)
        for( ssize_t sj=0 ; sj<subdiv ; sj++)
        for( ssize_t si=0 ; si<subdiv ; si++)
        {
          const ssize_t subcell_index = grid_ijk_to_index( IJK{subdiv,subdiv,subdiv} , IJK{si,sj,sk} );
          const ssize_t value_index = grid_cell_index * stride + subcell_index;
          assert( value_index >= 0 );          
          if( is_ghost )
          {
            cc_label_ptr[value_index] = ghost_no_label;
          }
          else
          {
            if( density_ptr[value_index] < threshold )
            {
              cc_label_ptr[value_index] = no_label;
            }
            else
            {
              const uint64_t unique_id = encode_unique_id( domain_cell_index, grid_cell_index, subcell_index );
              const double label = static_cast<double>(unique_id);
              assert( label == unique_id ); // ensures lossless conversion to double
              cc_label_ptr[value_index] = label;
              assert( owner_from_unique_id(unique_id) == rank );
            }
          }
        }
      }

      /****************************************************
       * === propagate_minimum_label ===
       * Propagates minimum CC label id from nearby cells.
       * When done, all MPI processes assigned the same
       * globaly known unique label ids to connected cells.
       ****************************************************/
      unsigned long long label_update_passes = 0;
      unsigned long long total_local_passes = 0;
      unsigned long long total_comm_passes = 0;
      do
      {
        ++ total_comm_passes;
        grid_update_ghosts( ::exanb::ldbg, *mpi, *ghost_comm_scheme, null_grid_ptr, *domain, grid_cell_values.get_pointer(),
                            *ghost_comm_buffers, pecfunc,pesfunc, update_fields,
                            0, false, false, false,
                            true, false , std::integral_constant<bool,false>{} );

        unsigned long long local_propagate_pass = 0;
        unsigned long long label_update_count = 0;
        label_update_passes = 0;
        do
        {
          ++ local_propagate_pass;
          ++ total_local_passes;

          label_update_count = 0;
#         pragma omp parallel for collapse(3) schedule(static) reduction(+:label_update_count)
          for( ssize_t k=gl ; k < (grid_dims.k-gl) ; k++)
          for( ssize_t j=gl ; j < (grid_dims.j-gl) ; j++)
          for( ssize_t i=gl ; i < (grid_dims.i-gl) ; i++)
          {        
            const IJK cell_loc = {i,j,k};
            const ssize_t cell_index = grid_ijk_to_index( grid_dims , cell_loc );
            for( ssize_t sk=0 ; sk<subdiv ; sk++)
            for( ssize_t sj=0 ; sj<subdiv ; sj++)
            for( ssize_t si=0 ; si<subdiv ; si++)
            {
              const IJK subcell_loc = {si,sj,sk};
              ssize_t subcell_index = grid_ijk_to_index( IJK{subdiv,subdiv,subdiv} , subcell_loc );
              ssize_t value_index = cell_index * stride + subcell_index;
              assert( value_index >= 0 );
              if( cc_label_ptr[ value_index ] >= 0.0 )
              {
                for( ssize_t nk=-1 ; nk<=1 ; nk++)
                for( ssize_t nj=-1 ; nj<=1 ; nj++)
                for( ssize_t ni=-1 ; ni<=1 ; ni++) if(ni!=0 || nj!=0 || nk!=0)
                {
                  IJK nbh_cell_loc={0,0,0}, nbh_subcell_loc={0,0,0};
                  gcv_subcell_neighbor( cell_loc, subcell_loc, subdiv, IJK{ni,nj,nk}, nbh_cell_loc, nbh_subcell_loc );
                  if( nbh_cell_loc.i>=0 && nbh_cell_loc.i<grid_dims.i
                   && nbh_cell_loc.j>=0 && nbh_cell_loc.j<grid_dims.j
                   && nbh_cell_loc.k>=0 && nbh_cell_loc.k<grid_dims.k )
                  {
                    // is neighbor in the ghost cell area ?
                    /* const bool nbh_is_ghost = (nbh_cell_loc.i<gl) || (nbh_cell_loc.i>=(grid_dims.i-gl))
                                           || (nbh_cell_loc.j<gl) || (nbh_cell_loc.j>=(grid_dims.j-gl))
                                           || (nbh_cell_loc.k<gl) || (nbh_cell_loc.k>=(grid_dims.k-gl)); */
                    const ssize_t nbh_cell_index = grid_ijk_to_index( grid_dims , nbh_cell_loc );
                    const ssize_t nbh_subcell_index = grid_ijk_to_index( IJK{subdiv,subdiv,subdiv} , nbh_subcell_loc );
                    const ssize_t nbh_value_index = nbh_cell_index * stride + nbh_subcell_index;
                    assert( nbh_value_index >= 0 );
                    // assert( cc_label_ptr[ nbh_value_index ] != ghost_no_label );
                    if( cc_label_ptr[nbh_value_index] >= 0.0 && cc_label_ptr[nbh_value_index] < cc_label_ptr[value_index] )
                    {
                      cc_label_ptr[ value_index ] = cc_label_ptr[ nbh_value_index ];
                      ++ label_update_count;
                    }
                  }
                }
              }
            }
          }
        
          if( label_update_count > 0 ) ++ label_update_passes;
        } while( label_update_count > 0 );

        MPI_Allreduce(MPI_IN_PLACE,&label_update_passes,1,MPI_UNSIGNED_LONG_LONG,MPI_MAX,*mpi);
        
        ldbg << "Max local label update passes = "<<label_update_passes<<std::endl;
      } while( label_update_passes > 0 );

      ldbg << "total_local_passes="<<total_local_passes<<" , total_comm_passes="<<total_comm_passes<<std::endl;


      /*******************************************************************
       * count number of local ids and identify their respective owner process
       *******************************************************************/
      std::unordered_map<size_t,ConnectedComponentInfo> cc_map;
      // const size_t own_label_count = 0;
      for( ssize_t k=gl ; k < (grid_dims.k-gl) ; k++)
      for( ssize_t j=gl ; j < (grid_dims.j-gl) ; j++)
      for( ssize_t i=gl ; i < (grid_dims.i-gl) ; i++)
      {        
        const IJK cell_loc = {i,j,k};
        const IJK domain_cell_loc = cell_loc + grid_offset ;
        const ssize_t cell_index = grid_ijk_to_index( grid_dims , cell_loc );
        for( ssize_t sk=0 ; sk<subdiv ; sk++)
        for( ssize_t sj=0 ; sj<subdiv ; sj++)
        for( ssize_t si=0 ; si<subdiv ; si++)
        {
          const IJK subcell_loc = {si,sj,sk};
          const ssize_t subcell_index = grid_ijk_to_index( IJK{subdiv,subdiv,subdiv} , subcell_loc );
          const ssize_t value_index = cell_index * stride + subcell_index;
          assert( value_index >= 0 );
          const double label = cc_label_ptr[ value_index ];
          if( label >= 0.0 )
          {
            const ssize_t unique_id = static_cast<size_t>( label );
            auto & cc_info = cc_map[unique_id];
            if( cc_info.m_label == -1.0 )
            {
              cc_info.m_label = label;
              cc_info.m_rank = -1; // not known yet
              cc_info.m_cell_count = 0;
              cc_info.m_center = Vec3d{0.,0.,0.};
              cc_info.m_gyration = Mat3d{ 0.,0.,0., 0.,0.,0., 0.,0.,0. };
            }
            else
            {
              assert( cc_info.m_label == label );
            }
            cc_info.m_cell_count += 1;
            cc_info.m_center += make_vec3d( ( domain_cell_loc * subdiv ) + subcell_loc );
            // cc_info.m_gyration += ... ;
          }
        }
      }

      // update map assigning correct destination process in m_rank field
      for(auto & cc : cc_map)
      {
        assert( cc.second.m_label >= 0.0 );
        unsigned long long unique_id = static_cast<unsigned long long>( cc.second.m_label );
        cc.second.m_rank = owner_from_unique_id( unique_id );
        assert( cc.second.m_rank >= 0 && cc.second.m_rank < nprocs );
      }

      // count how many cc table entries we'll send to each partner process
      ldbg << "cc_map.size() = "<<cc_map.size()<<std::endl;
      std::vector<int> cc_send_counts( nprocs , 0 );
      std::vector<int> cc_recv_counts( nprocs , 0 );
      for(const auto & cc : cc_map)
      {
        const int dest_proc = cc.second.m_rank;
        assert( dest_proc>=0 && dest_proc<nprocs);
        cc_send_counts[ dest_proc ] += 1;
      }

      // here, we know only how many elements we'll send to others processes,
      // the following communication allows to get the number of elements we will receive from others
      MPI_Alltoall( cc_send_counts.data() , 1 , MPI_INT , cc_recv_counts.data() , 1 , MPI_INT , *mpi );      
      std::vector<int> cc_send_displs( nprocs , 0 );
      std::vector<int> cc_recv_displs( nprocs , 0 );
      size_t cc_total_send = 0;
      size_t cc_total_recv = 0;
      for(int i=0;i<nprocs;i++)
      {
        cc_send_displs[i] = cc_total_send;
        cc_total_send += cc_send_counts[i];
        cc_recv_displs[i] = cc_total_recv;
        cc_total_recv += cc_recv_counts[i];
        // ldbg << "SEND["<<i<<"] : c="<<cc_send_counts[i]<<" d="<<cc_send_displs[i]<<std::endl;
        // ldbg << "RECV["<<i<<"] : c="<<cc_recv_counts[i]<<" d="<<cc_recv_displs[i]<<std::endl;
      }
      assert( cc_total_send == cc_map.size() );
      ldbg << "cc_total_send="<<cc_total_send<<" , cc_total_recv="<<cc_total_recv<<std::endl; 
      
      
      std::vector<ConnectedComponentInfo> cc_recv_data( cc_total_recv , ConnectedComponentInfo{} );
      std::vector<ConnectedComponentInfo> cc_send_data( cc_total_send , ConnectedComponentInfo{} );      
      // fill send buffer from map ith respect to process rank order
      cc_total_send = 0;
      for(const auto & cc : cc_map)
      {
        const int dest_proc = cc.second.m_rank;
        assert( dest_proc>=0 && dest_proc<nprocs);
        assert( cc_send_data[cc_send_displs[dest_proc]].m_label == -1.0 );
        cc_send_data[ cc_send_displs[dest_proc] ++ ] = cc.second;
      }
      cc_map.clear();

#     ifndef NDEBUG
      // make sur there's no hole left
      for(const auto& cc:cc_send_data) { assert( cc.m_label >= 0.0 ); }
#     endif

      for(int i=0;i<nprocs;i++)
      {
        cc_send_displs[i] -= cc_send_counts[i];
        cc_send_counts[i] *= sizeof(ConnectedComponentInfo);
        cc_recv_counts[i] *= sizeof(ConnectedComponentInfo);
        cc_send_displs[i] *= sizeof(ConnectedComponentInfo);
        cc_recv_displs[i] *= sizeof(ConnectedComponentInfo);
        // ldbg << "* SEND["<<i<<"] : c="<<cc_send_counts[i]/sizeof(ConnectedComponentInfo)<<" d="<<cc_send_displs[i]/sizeof(ConnectedComponentInfo)<<std::endl;
        // ldbg << "* RECV["<<i<<"] : c="<<cc_recv_counts[i]/sizeof(ConnectedComponentInfo)<<" d="<<cc_recv_displs[i]/sizeof(ConnectedComponentInfo)<<std::endl;
      }
      MPI_Alltoallv( cc_send_data.data() , cc_send_counts.data() , cc_send_displs.data() , MPI_BYTE
                   , cc_recv_data.data() , cc_recv_counts.data() , cc_recv_displs.data() , MPI_BYTE
                   , *mpi );

      cc_send_counts.clear(); cc_send_counts.shrink_to_fit();
      cc_recv_counts.clear(); cc_recv_counts.shrink_to_fit();
      cc_send_displs.clear(); cc_send_displs.shrink_to_fit();
      cc_recv_displs.clear(); cc_recv_displs.shrink_to_fit();
      cc_send_data.clear(); cc_send_data.shrink_to_fit();

      
      /*************************************************************************
       * finalize CC information statistics and filter out some of the CCs
       * dependending on optional filtering parameters.
       *************************************************************************/
      for(const auto & cc : cc_recv_data)
      {
        const ssize_t unique_id = static_cast<size_t>( cc.m_label );
        auto & cc_info = cc_map[unique_id];
        if( cc_info.m_label == -1.0 )
        {
          cc_info.m_label = cc.m_label;
          cc_info.m_rank = cc.m_rank;
          cc_info.m_cell_count = 0;
          cc_info.m_center = Vec3d{0.,0.,0.};
          cc_info.m_gyration = Mat3d{ 0.,0.,0., 0.,0.,0., 0.,0.,0. };
        }
        else
        {
          assert( cc_info.m_label == cc.m_label );
          assert( cc_info.m_rank == cc.m_rank );
          assert( cc_info.m_rank == rank );
        }
        cc_info.m_cell_count += cc.m_cell_count;
        cc_info.m_center += cc.m_center;
        cc_info.m_gyration += cc.m_gyration;
      }

      cc_table->m_table.clear();
      for(const auto & ccp : cc_map)
      {
        if( ccp.second.m_cell_count >= (*cc_count_threshold) )
        {
          cc_table->m_table.push_back( ccp.second );
          assert( cc_table->m_table.back().m_rank == rank );
          cc_table->m_table.back().m_rank = -1;
        }
      }
      cc_map.clear();
      std::sort( cc_table->m_table.begin() , cc_table->m_table.end() , []( const ConnectedComponentInfo& cc_a , const ConnectedComponentInfo& cc_b ) -> bool { return cc_a.m_label < cc_b.m_label; } );

      unsigned long long global_label_idx_start = 0;
      unsigned long long global_label_count = cc_table->m_table.size();
      MPI_Exscan( &global_label_count , &global_label_idx_start , 1 , MPI_UNSIGNED_LONG_LONG , MPI_SUM , *mpi );
      MPI_Allreduce( MPI_IN_PLACE , &global_label_count , 1 , MPI_UNSIGNED_LONG_LONG , MPI_SUM , *mpi );
      cc_table->m_global_label_count = global_label_count;
      cc_table->m_local_label_start = global_label_idx_start;

      // (!) starting from here, m_rank field is not the rank of owning processor any more,
      // but the global label index, in [0;number_of_ccs[, of CC
      for(size_t i=0;i<cc_table->size();i++)
      {
        cc_table->at(i).m_rank = i + global_label_idx_start;
        assert( cc_table->at(i).m_label >= 0.0 );
        const uint64_t unique_id = strip_unique_id( static_cast<uint64_t>( cc_table->at(i).m_label ) );
        cc_table->at(i).m_label = unique_id;
        assert( unique_id == cc_table->at(i).m_label );
        // ldbg << "CC["<<cc_table->at(i).m_rank<<"] : id="<<unique_id<<", count="<<cc_table->at(i).m_cell_count<<std::endl;
      }

      // also strip unique_id (cc_label) stored in cell values
#     pragma omp parallel for collapse(3) schedule(static)
      for( ssize_t k=gl ; k < (grid_dims.k-gl) ; k++)
      for( ssize_t j=gl ; j < (grid_dims.j-gl) ; j++)
      for( ssize_t i=gl ; i < (grid_dims.i-gl) ; i++)
      {        
        const ssize_t cell_index = grid_ijk_to_index( grid_dims , IJK{i,j,k} );
        for( ssize_t sk=0 ; sk<subdiv ; sk++)
        for( ssize_t sj=0 ; sj<subdiv ; sj++)
        for( ssize_t si=0 ; si<subdiv ; si++)
        {
          const ssize_t subcell_index = grid_ijk_to_index( IJK{subdiv,subdiv,subdiv} , IJK{si,sj,sk} );
          const ssize_t value_index = cell_index * stride + subcell_index;
          assert( value_index >= 0 );
          if( cc_label_ptr[value_index] >= 0.0 ) cc_label_ptr[value_index] = strip_unique_id( static_cast<uint64_t>( cc_label_ptr[value_index] ) );              
        }
      }

      ldbg << "cc_label : owned_label_count="<<cc_table->size()<<", global_label_count="<<global_label_count<<", global_label_idx_start="<<global_label_idx_start << std::endl;
    }

    // -----------------------------------------------
    // -----------------------------------------------
    inline std::string documentation() const override final
    {
      return R"EOF(
Computes cell connected components information
)EOF";
    }

  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(cc_label)
  {
   OperatorNodeFactory::instance()->register_factory("cc_label", make_simple_operator< ConnectedComponentLabel > );
  }

}
