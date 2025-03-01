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
#include <bit>

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
      
      assert( domain_dims.i>=0 && domain_dims.j>=0 && domain_dims.k>=0 );
      const unsigned int unique_id_i_bits = std::bit_width( size_t(domain_dims.i) );
      const unsigned int unique_id_j_bits = std::bit_width( size_t(domain_dims.j) );
      const unsigned int unique_id_k_bits = std::bit_width( size_t(domain_dims.k) );
      const unsigned int unique_id_total_bits = unique_id_i_bits + unique_id_j_bits + unique_id_k_bits;
      const uint64_t unique_id_count = ( 1ull << unique_id_total_bits ) * subdiv3;

      // these 3 funcions define how we build unique ids such a way that :
      // 1. owner rank can be guessed from the unique_id
      // 2. final (stripped) unique ids will be deterministic and independant from parallel settings
      auto encode_unique_id = [&](const IJK& domain_cell_loc, uint64_t subcell_index) -> uint64_t
      {
        assert(rank>=0 && rank<nprocs);
        unsigned int i_bits = unique_id_i_bits;
        unsigned int j_bits = unique_id_j_bits;
        unsigned int k_bits = unique_id_k_bits;
        unsigned int total_bits = unique_id_total_bits;
        uint64_t domain_cell_z_index = 0;
        while( total_bits > 0 )
        {
          unsigned int max_coord_bits = std::max( std::max( i_bits , j_bits ) , k_bits );
          assert( max_coord_bits > 0 );
          if( i_bits == max_coord_bits ) { assert(total_bits>0 && i_bits>0); --i_bits; --total_bits; domain_cell_z_index |= ( ( domain_cell_loc.i >> i_bits ) & 1ull ) << total_bits; }
          if( j_bits == max_coord_bits ) { assert(total_bits>0 && j_bits>0); --j_bits; --total_bits; domain_cell_z_index |= ( ( domain_cell_loc.j >> j_bits ) & 1ull ) << total_bits; }
          if( k_bits == max_coord_bits ) { assert(total_bits>0 && k_bits>0); --k_bits; --total_bits; domain_cell_z_index |= ( ( domain_cell_loc.k >> k_bits ) & 1ull ) << total_bits; }
        }
        assert( total_bits==0 && i_bits==0 && j_bits==0 && k_bits==0 );
        //uint64_t domain_cell_index = grid_ijk_to_index( domain_dims , domain_cell_loc );
        return ( domain_cell_z_index * subdiv3 + subcell_index ) ; //* nprocs + rank;
      };
            
      // now we can build a function to determine final destination (process rank) for each CC based on its label
      unsigned long long unique_id_min = unique_id_count;
      unsigned long long unique_id_max = 0;
      auto owner_from_unique_id = [&](uint64_t unique_id) -> int 
      {
        assert( unique_id_min < unique_id_max );
        const unsigned long long range_count = ( unique_id_max - unique_id_min ) / subdiv3;
        const unsigned long long id_balance_offset = ( range_count / nprocs ) / 2;
        const unsigned long long id = ( ( unique_id - unique_id_min ) / subdiv3 ) + id_balance_offset;
        int p = ( range_count > 0 ) ? ( ( id * nprocs ) / range_count ) : 0 ;
        return std::clamp( p , 0 , nprocs-1 );
      };

      // create a unique label id for each cell satisfying selection criteria
#     pragma omp parallel for collapse(3) schedule(static)
      for( ssize_t k=0 ; k < grid_dims.k ; k++)
      for( ssize_t j=0 ; j < grid_dims.j ; j++)
      for( ssize_t i=0 ; i < grid_dims.i ; i++)
      {
        const bool is_ghost = (i<gl) || (i>=(grid_dims.i-gl)) || (j<gl) || (j>=(grid_dims.j-gl)) || (k<gl) || (k>=(grid_dims.k-gl));
        const IJK grid_cell_loc = {i,j,k};
        const IJK domain_cell_loc = grid_cell_loc + grid_offset;
        const ssize_t grid_cell_index = grid_ijk_to_index( grid_dims , grid_cell_loc );
        for( ssize_t sk=0 ; sk<subdiv ; sk++)
        for( ssize_t sj=0 ; sj<subdiv ; sj++)
        for( ssize_t si=0 ; si<subdiv ; si++)
        {
          const ssize_t subcell_index = grid_ijk_to_index( IJK{subdiv,subdiv,subdiv} , IJK{si,sj,sk} );
          const ssize_t value_index = grid_cell_index * stride + subcell_index;
          assert( value_index >= 0 );          
          if( is_ghost )
          {
            cc_label_ptr[value_index] = ConnectedComponentInfo::GHOST_NO_LABEL;
          }
          else
          {
            if( density_ptr[value_index] < threshold )
            {
              cc_label_ptr[value_index] = ConnectedComponentInfo::NO_LABEL;
            }
            else
            {
              const uint64_t unique_id = encode_unique_id( domain_cell_loc, subcell_index );
              const double label = static_cast<double>(unique_id);
              assert( label == unique_id ); // ensures lossless conversion to double
              cc_label_ptr[value_index] = label;
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
      std::unordered_map<uint64_t,uint64_t> id_fast_remap(4096);
      do
      {
        grid_update_ghosts( ::exanb::ldbg, *mpi, *ghost_comm_scheme, null_grid_ptr, *domain, grid_cell_values.get_pointer(),
                            *ghost_comm_buffers, pecfunc,pesfunc, update_fields,
                            0, false, false, false,
                            true, false , std::integral_constant<bool,false>{} );

        unsigned long long label_update_count = 0;
        label_update_passes = 0;
        do
        {
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
              double old_value = ConnectedComponentInfo::NO_LABEL;
#             pragma omp atomic read
              old_value = cc_label_ptr[ value_index ];
              if( old_value >= 0.0 )
              {
                double new_value = old_value;
                const uint64_t old_unique_id = static_cast<uint64_t>( old_value );
                auto remap_it = id_fast_remap.find(old_unique_id);
                double remap_label = ConnectedComponentInfo::NO_LABEL;
                if( remap_it != id_fast_remap.end() )
                {
#                 pragma omp atomic read
                  remap_label = remap_it->second;
                }
                if( remap_label != ConnectedComponentInfo::NO_LABEL && remap_label < new_value )
                {
                  new_value = remap_label;
                }

                for(int ni=-1;ni<=1;ni++)
                for(int nj=-1;nj<=1;nj++)
                for(int nk=-1;nk<=1;nk++)
                if(ni!=0||nj!=0||nk!=0)
                {
                  const IJK nbh_ijk = {ni,nj,nk};
                  IJK nbh_cell_loc={0,0,0}, nbh_subcell_loc={0,0,0};
                  gcv_subcell_neighbor( cell_loc, subcell_loc, subdiv, nbh_ijk, nbh_cell_loc, nbh_subcell_loc );
                  if( nbh_cell_loc.i>=0 && nbh_cell_loc.i<grid_dims.i
                   && nbh_cell_loc.j>=0 && nbh_cell_loc.j<grid_dims.j
                   && nbh_cell_loc.k>=0 && nbh_cell_loc.k<grid_dims.k )
                  {
                    const ssize_t nbh_cell_index = grid_ijk_to_index( grid_dims , nbh_cell_loc );
                    const ssize_t nbh_subcell_index = grid_ijk_to_index( IJK{subdiv,subdiv,subdiv} , nbh_subcell_loc );
                    const ssize_t nbh_value_index = nbh_cell_index * stride + nbh_subcell_index;
                    assert( nbh_value_index >= 0 );
                    double nbh_cc_label = ConnectedComponentInfo::NO_LABEL;
#                   pragma omp atomic read
                    nbh_cc_label = cc_label_ptr[nbh_value_index];
                    if( nbh_cc_label >= 0.0 && nbh_cc_label < new_value )
                    {
                      new_value = nbh_cc_label;
                    }
                  }
                }
                if( new_value < old_value )
                {
                  ++ label_update_count;
                  if( new_value < remap_label )
                  {
#                   pragma omp atomic write
                    remap_it->second = new_value;
                  }
#                 pragma omp atomic write
                  cc_label_ptr[value_index] = new_value;
                }
              }
            }
          }
        
          if( label_update_count > 0 ) ++ label_update_passes;

        } while( label_update_count > 0 );

        MPI_Allreduce(MPI_IN_PLACE,&label_update_passes,1,MPI_UNSIGNED_LONG_LONG,MPI_MAX,*mpi);
        ldbg << "Max local label update passes = "<<label_update_passes<<std::endl;
        
        if( label_update_passes > 0 )
        {
          id_fast_remap.clear();
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
              const double label = cc_label_ptr[ value_index ];
              if( label >= 0.0 )
              {
                const uint64_t unique_id = static_cast<uint64_t>( label );
                if( id_fast_remap.find(unique_id) == id_fast_remap.end() )
                {
                  id_fast_remap.insert( {unique_id,unique_id} );
                }
              }
            }
          }
        }
        
        ++ total_comm_passes;
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
            unique_id_min = std::min( unique_id_min , static_cast<unsigned long long>( unique_id ) );
            unique_id_max = std::max( unique_id_max , static_cast<unsigned long long>( unique_id+1 ) );
            auto & cc_info = cc_map[unique_id];
            if( cc_info.m_label == ConnectedComponentInfo::NO_LABEL )
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

      MPI_Allreduce(MPI_IN_PLACE,&unique_id_min,1,MPI_UNSIGNED_LONG_LONG,MPI_MIN,*mpi);
      MPI_Allreduce(MPI_IN_PLACE,&unique_id_max,1,MPI_UNSIGNED_LONG_LONG,MPI_MAX,*mpi);
      ldbg << "id min = "<<unique_id_min<<" , id max = "<<unique_id_max <<std::endl;

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
        assert( cc_send_data[cc_send_displs[dest_proc]].m_label == ConnectedComponentInfo::NO_LABEL );
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
      }
      MPI_Alltoallv( cc_send_data.data() , cc_send_counts.data() , cc_send_displs.data() , MPI_BYTE
                   , cc_recv_data.data() , cc_recv_counts.data() , cc_recv_displs.data() , MPI_BYTE
                   , *mpi );
      
      /*************************************************************************
       * finalize CC information statistics and filter out some of the CCs
       * dependending on optional filtering parameters.
       *************************************************************************/
      for(const auto & cc : cc_recv_data)
      {
        const ssize_t unique_id = static_cast<size_t>( cc.m_label );
        auto & cc_info = cc_map[unique_id];
        if( cc_info.m_label == ConnectedComponentInfo::NO_LABEL )
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

      unsigned long long global_label_count = 0;
      std::set<int64_t> ordered_label_ids;
      for(auto & ccp : cc_map)
      {
        ordered_label_ids.insert( ccp.first );
        if( ccp.second.m_cell_count < (*cc_count_threshold) ) ccp.second.m_cell_count = 0;
        else ++ global_label_count;
      }

      unsigned long long global_label_idx_start = 0;
      MPI_Exscan( &global_label_count , &global_label_idx_start , 1 , MPI_UNSIGNED_LONG_LONG , MPI_SUM , *mpi );
      MPI_Allreduce( MPI_IN_PLACE , &global_label_count , 1 , MPI_UNSIGNED_LONG_LONG , MPI_SUM , *mpi );

      unsigned long long global_label_idx_end = global_label_idx_start;
      std::unordered_map<int64_t,int64_t> final_label_id_map;
      for(const auto unique_id : ordered_label_ids)
      {
        auto & cc = cc_map[unique_id];
        if( cc.m_cell_count > 0 )
        {
          cc.m_rank = global_label_idx_end ++;
        }
        else
        {
          cc.m_rank = -1;
        }
        final_label_id_map[unique_id] = cc.m_rank;
      }

      // fill back recv_data with updated data so sender can now have complete information
      // about CCs it does not own
      for(auto & cc : cc_recv_data)
      {
        const ssize_t unique_id = static_cast<size_t>( cc.m_label );
        cc = cc_map[unique_id];
      }
      
      // reverse communication to send back cc info update
      MPI_Alltoallv( cc_recv_data.data() , cc_recv_counts.data() , cc_recv_displs.data() , MPI_BYTE
                   , cc_send_data.data() , cc_send_counts.data() , cc_send_displs.data() , MPI_BYTE
                   , *mpi );

      for(const auto & cc : cc_send_data)
      {
        const ssize_t unique_id = static_cast<size_t>( cc.m_label );
        final_label_id_map[unique_id] = cc.m_rank;
      }

      cc_table->m_table.clear();
      for(const auto unique_id : ordered_label_ids)
      {
        auto & cc = cc_map[unique_id];
        if( cc.m_cell_count > 0 )
        {
          assert( cc.m_rank>=global_label_idx_start && cc.m_rank<global_label_idx_end );
          assert( ( cc.m_rank - global_label_idx_start ) == cc_table->m_table.size() );
          cc.m_label = cc.m_rank;
          cc.m_rank = rank;
          cc_table->m_table.push_back( cc );
        }
      }
      cc_map.clear();

      // update cc_label grid cell values with final label ids
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
          const ssize_t subcell_index = grid_ijk_to_index( IJK{subdiv,subdiv,subdiv} , subcell_loc );
          const ssize_t value_index = cell_index * stride + subcell_index;
          assert( value_index >= 0 );
          const double label = cc_label_ptr[ value_index ];
          if( label >= 0.0 )
          {
            const ssize_t unique_id = static_cast<size_t>( label );
            auto it = final_label_id_map.find( unique_id );
            assert( it != final_label_id_map.end() );
            cc_label_ptr[ value_index ] = it->second;
          }
          else
          {
            cc_label_ptr[ value_index ] = ConnectedComponentInfo::NO_CC_LABEL;
          }
        }
      }

      ldbg << "cc_label : owned_label_count="<<cc_table->size()<<", global_label_count="<<global_label_count
           <<", global_label_idx_start="<<global_label_idx_start <<", global_label_idx_end="<<global_label_idx_end << std::endl;
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
