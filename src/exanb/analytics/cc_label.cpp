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

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/domain.h>
#include <exanb/grid_cell_particles/grid_cell_values.h>
#include <exanb/core/grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/mpi/grid_update_ghosts.h>
#include <exanb/analytics/cc_info.h>

#include <mpi.h>
#include <onika/silent_use.h>

namespace exanb
{
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
      const double subcell_volume = subcell_size * subcell_size * subcell_size;
      
      // dimension of the subdivided simulation's grid
      const IJK domain_subdiv_dims = domain_dims * subdiv; 

      // some debug information
      ldbg << "ghost_layers="<<gl<<", cell_size="<<cell_size<<", subdiv="<<subdiv<<", subcell_size="
           <<subcell_size<<", grid_dims="<<grid_dims<<", grid_offset="<<grid_offset<<", domain_dims="<<domain_dims<<", domain_subdiv_dims="<<domain_subdiv_dims<<std::endl;

      // note: if we are to add new data fields, they must be added BEFORE we retreive access information

      // create additional data field for connected component label
      if( ! grid_cell_values->has_field("cc_label") )
      {
        grid_cell_values->add_field("cc_label",subdiv,1);
      }
      
      // retreive cc_label field data accessor.
      const auto cc_label_accessor = grid_cell_values->field_data("cc_label");
      double * __restrict__ cc_label_ptr = cc_label_accessor.m_data_ptr;
      const size_t cc_label_stride = cc_label_accessor.m_stride; ONIKA_SILENT_USE(cc_label_stride);

      // retreive density field data accessor.
      const auto density_accessor = grid_cell_values->field_data( *grid_cell_field );
      const double  * __restrict__ density_ptr = density_accessor.m_data_ptr;
      const size_t density_stride = density_accessor.m_stride;
      
      // sanity check
      assert( cc_label_stride == density_stride );
      const size_t stride = density_stride; // for simplicity

      const double threshold = *grid_cell_threshold ;

      // as a preamble, build a ghost cell owner map
      int rank=0, nprocs=1;
      MPI_Comm_rank( *mpi , &rank );
      MPI_Comm_size( *mpi , &nprocs );
      std::unordered_map<ssize_t,int> cell_owner_rank;
      for(int p=0;p<nprocs;p++)
      {
        if( p != rank )
        {
          const size_t n_cells_to_receive = ghost_comm_scheme->m_partner[p].m_receives.size();
          const auto * __restrict__ recvs = ghost_comm_scheme->m_partner[p].m_receives.data();
          for(size_t i=0;i<n_cells_to_receive;i++)
          {
            const auto recv_info = ghost_cell_receive_info( recvs[i] );
            cell_owner_rank[ recv_info.m_cell_i ] = p;
          }
        }
      }
      
      auto cell_owner = [&]( const IJK& loc ) -> int
      {
        ssize_t cell_idx = grid_ijk_to_index( grid_dims , loc );
        auto it = cell_owner_rank.find( cell_idx );
        if( it != cell_owner_rank.end() ) return it->second;
        else return rank;
      };

      // create a unique label id for each cell satisfying selction criteria
#     pragma omp parallel for collapse(3) schedule(static)
      for( ssize_t k=0 ; k < grid_dims.k ; k++)
      for( ssize_t j=0 ; j < grid_dims.j ; j++)
      for( ssize_t i=0 ; i < grid_dims.i ; i++)
      {
        // position of the cell in the simulation grid, which size is 'domain_dims'
        IJK cell_location = IJK{i,j,k} + grid_offset;

        // triple loop to enumerate sub cells inside a cell
        for( ssize_t sk=0 ; sk<subdiv ; sk++)
        for( ssize_t sj=0 ; sj<subdiv ; sj++)
        for( ssize_t si=0 ; si<subdiv ; si++)
        {
          // location of the subcell in simulation's subcell grid, which dimension is 'domain_subdiv_dims'
          IJK subcell_global_location = cell_location * subdiv + IJK{si,sj,sk};

          if( domain->periodic_boundary_x() && subcell_global_location.i <  0                    ) subcell_global_location.i += domain_subdiv_dims.i;
          if( domain->periodic_boundary_x() && subcell_global_location.i >= domain_subdiv_dims.i ) subcell_global_location.i -= domain_subdiv_dims.i;

          if( domain->periodic_boundary_y() && subcell_global_location.j <  0                    ) subcell_global_location.j += domain_subdiv_dims.j;
          if( domain->periodic_boundary_y() && subcell_global_location.j >= domain_subdiv_dims.j ) subcell_global_location.j -= domain_subdiv_dims.j;

          if( domain->periodic_boundary_z() && subcell_global_location.k <  0                    ) subcell_global_location.k += domain_subdiv_dims.k;
          if( domain->periodic_boundary_z() && subcell_global_location.k >= domain_subdiv_dims.k ) subcell_global_location.k -= domain_subdiv_dims.k;

          ssize_t subcell_global_index = grid_ijk_to_index( domain_subdiv_dims , subcell_global_location );
          assert( subcell_global_index >= 0 );

          // compute a simulation wide, processor invariant, sub cell label id;
          double label = static_cast<double>(subcell_global_index);

          // computation of subcell index in local processor's grid
          ssize_t cell_index = grid_ijk_to_index( grid_dims , IJK{i,j,k} );
          ssize_t subcell_index = grid_ijk_to_index( IJK{subdiv,subdiv,subdiv} , IJK{si,sj,sk} );

          // value index, is the index of current subcell for local processor's grid
          ssize_t value_index = cell_index * stride + subcell_index;
          assert( value_index >= 0 );
          
          // assign a label to cells where density is above given threshold, otherwise assign -1
          cc_label_ptr[ value_index ] = density_ptr[ value_index ] > threshold ? label : -1.0;
        }
      }

      auto pecfunc = [self=this](auto ... args) { return self->parallel_execution_context(args ...); };
      auto pesfunc = [self=this](unsigned int i) { return self->parallel_execution_stream(i); };
      Grid< FieldSet<> > * null_grid_ptr = nullptr;
      onika::FlatTuple<> update_fields = {};

      unsigned long long total_label_count = 0;
      unsigned long long max_local_labels = 0;
      std::unordered_map<ssize_t,ssize_t> label_id_transition;

      // identify unique labels owned by this MPI process
      unsigned long long local_id_start=0, local_id_end=0;
      std::unordered_map<double,unsigned long long> local_labels;

      /****************************************************
       * >>> propagate_minimum_label( pass_number ) <<<
       * Propagates minimum CC label id from nearby cells.
       * When done, all MPI processes assigned the same
       * globaly known unique label ids to connected cells.
       ****************************************************/
      auto propagate_minimum_label = [&](int pass_number)
      {      
        unsigned long long label_update_passes = 0;   
        total_label_count = 0;
        label_id_transition.clear();

        do
        {
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
            //ldbg << "local propagate pass "<< local_propagate_pass << std::endl;
            label_update_count = 0;

#           pragma omp parallel for collapse(3) schedule(static) reduction(+:label_update_count)
            for( ssize_t k=0 ; k < grid_dims.k ; k++)
            for( ssize_t j=0 ; j < grid_dims.j ; j++)
            for( ssize_t i=0 ; i < grid_dims.i ; i++)
            {        
              const IJK cell_loc = {i,j,k};
              for( ssize_t sk=0 ; sk<subdiv ; sk++)
              for( ssize_t sj=0 ; sj<subdiv ; sj++)
              for( ssize_t si=0 ; si<subdiv ; si++)
              {
                const IJK subcell_loc = {si,sj,sk};
                ssize_t cell_index = grid_ijk_to_index( grid_dims , cell_loc );
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
                      ssize_t nbh_cell_index = grid_ijk_to_index( grid_dims , nbh_cell_loc );
                      ssize_t nbh_subcell_index = grid_ijk_to_index( IJK{subdiv,subdiv,subdiv} , nbh_subcell_loc );
                      ssize_t nbh_value_index = nbh_cell_index * stride + nbh_subcell_index;
                      assert( nbh_value_index >= 0 );
                      if( cc_label_ptr[ nbh_value_index ] >= 0.0 && cc_label_ptr[ nbh_value_index ] < cc_label_ptr[ value_index ] )
                      {
                        if( pass_number > 1 )
                        {
                          assert( max_local_labels > 0 );
                          ssize_t prev_label_id = static_cast<ssize_t>( cc_label_ptr[value_index] );
                          ssize_t new_label_id = static_cast<ssize_t>( cc_label_ptr[nbh_value_index] );
                          int prev_owner_rank = prev_label_id / max_local_labels;
                          int new_owner_rank = new_label_id / max_local_labels;
                          assert( new_owner_rank < prev_owner_rank );
#                         pragma omp critical(label_id_map_update)
                          label_id_transition[ prev_owner_rank ] = new_owner_rank;
                        }
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
      };

      /********************************************
       * >>> compact_label_ids( pass_number ) <<<
       * Reduces the range of label ids used.
       * Number of label ids used on each MPI process is equal to the number
       * of CC traversing process's sub-domain.
       * At the end of this function, label ids diverge accross MPI processes
       * as renumbering is done locally on each process.
       * Nevertheless, local renumbering takes into account number of local ids
       * on other processes such that local ids used by MPI processes are globally
       * not intersecting.
       ********************************************/
      auto compact_label_ids = [&](int pass_number) -> bool
      {
        bool labels_updated = false;
        local_labels.clear();
        for( ssize_t k=0 ; k < grid_dims.k ; k++)
        for( ssize_t j=0 ; j < grid_dims.j ; j++)
        for( ssize_t i=0 ; i < grid_dims.i ; i++)
        {        
          // local MPI sub domain cell location
          const IJK cell_loc = {i,j,k};

          // position of the cell in the simulation grid, which size is 'domain_dims'
          const IJK domain_cell_loc = cell_loc + grid_offset;

          for( ssize_t sk=0 ; sk<subdiv ; sk++)
          for( ssize_t sj=0 ; sj<subdiv ; sj++)
          for( ssize_t si=0 ; si<subdiv ; si++)
          {
            const IJK subcell_loc = {si,sj,sk};
            const ssize_t cell_index = grid_ijk_to_index( grid_dims , cell_loc );
            const ssize_t subcell_index = grid_ijk_to_index( IJK{subdiv,subdiv,subdiv} , subcell_loc );
            const ssize_t value_index = cell_index * stride + subcell_index;
            assert( value_index >= 0 );
            const double label = cc_label_ptr[ value_index ];
            if( label >= 0.0 )
            {
              if( ( pass_number == 1 ) || ( label >= local_id_start && label < local_id_end ) )
              {
                if( local_labels.find(label) == local_labels.end() ) local_labels.insert( { label , local_labels.size() } );
              }
            }
          }
        }
        
        // number of locally known CC labels
        total_label_count = local_labels.size();        
        max_local_labels = total_label_count;
        MPI_Allreduce( MPI_IN_PLACE , &max_local_labels , 1 , MPI_UNSIGNED_LONG_LONG , MPI_MAX , *mpi );
        
        // CC label id rane assigned to this MPI process
        local_id_start = rank * max_local_labels ;
        local_id_end = local_id_start + total_label_count;
        //MPI_Exscan( &total_label_count , &local_id_start , 1 , MPI_UNSIGNED_LONG_LONG , MPI_SUM , *mpi );
        //local_id_end = local_id_start + total_label_count;

        // sum of all locally known labels (greater than number of unique labels)
        MPI_Allreduce( MPI_IN_PLACE , &total_label_count , 1 , MPI_UNSIGNED_LONG_LONG , MPI_SUM , *mpi );
        
        // update local label ids in local id map
        for( auto & cc : local_labels ) { cc.second += local_id_start; }

        ldbg << "local_labels.size()="<<local_labels.size()<<" , total_label_count="<< total_label_count
             <<", local_id_start="<<local_id_start <<", local_id_end="<<local_id_end <<std::endl;

        for( ssize_t k=0 ; k < grid_dims.k ; k++)
        for( ssize_t j=0 ; j < grid_dims.j ; j++)
        for( ssize_t i=0 ; i < grid_dims.i ; i++)
        {        
          const IJK cell_loc = {i,j,k};
          for( ssize_t sk=0 ; sk<subdiv ; sk++)
          for( ssize_t sj=0 ; sj<subdiv ; sj++)
          for( ssize_t si=0 ; si<subdiv ; si++)
          {
            const IJK subcell_loc = {si,sj,sk};
            const ssize_t cell_index = grid_ijk_to_index( grid_dims , cell_loc );
            const ssize_t subcell_index = grid_ijk_to_index( IJK{subdiv,subdiv,subdiv} , subcell_loc );
            const ssize_t value_index = cell_index * stride + subcell_index;
            assert( value_index >= 0 );
            const double label = cc_label_ptr[ value_index ];
            if( label >= 0.0 )
            {
              auto it = local_labels.find( label );
              if( it != local_labels.end() )
              {
                const double new_label = static_cast<double>( it->second );
                if( new_label != label )
                {
                  labels_updated = true;
                  cc_label_ptr[ value_index ] = new_label;
                }
              }
            }
          }
        }
        return labels_updated;
      };
      
      
      /**************************************************************************************************
       * 1st pass :
       * ==========
       * 1.a) connect cells and build globally unique label ids for entire CC
       * accross MPI processes
       * 1.b) reduces the range of label ids,
       * label ids diverges again across MPI processes at the end.
       * Local renumbering guarantees label ids are different across MPI processes.
       *
       * 2nd pass :
       * ==========
       * 2.a) reconciliate reduced local ids by assigning local id of CC's
       * owner MPI rank to the entire CC across MPI processes.
       * during label propagation, temporary local ids from non owner processes are overriden
       * by label id of owner process. At the same time, a transition map is built to keep track
       * of how label ids have been transformed to their owner rank's local label id cunter part.
       * 2.b) during the 2nd pass, we count the number of locally owned CCs present in the sub-domain.
       * local_labels id map contains only entries for ids owned by local MPI process
       * during this 2nd pass.
       **************************************************************************************************/
      ldbg << "*** PASS 1 ***" << std::endl;
      propagate_minimum_label( 1 );
      compact_label_ids( 1 );
      ldbg << "*** PASS 2 ***" << std::endl;
      propagate_minimum_label( 2 );
      local_labels.clear(); // not needed anymore

      /**********************************************************************************************
       * performs 3 different tasks using a single traversal of label_id_transition values :
       * 1) transform destination label ids to their final value (following the chain of transitions)
       * => transitive closure of label id transition graph
       * 2) build local cc_table label to index map
       * 3) store what CC info entries has to be sent to foreign MPI processes
       **********************************************************************************************/
      ssize_t local_cc_table_size = local_id_end - local_id_start;
      ldbg << "total_label_count="<< total_label_count<< " , local_cc_table_size="<<local_cc_table_size<<" , local_id_start="<<local_id_start <<std::endl;
      cc_table->assign( local_cc_table_size , ConnectedComponentInfo{} );
      std::unordered_map<ssize_t,ssize_t> label_idx_map;
      std::unordered_map< int , std::unordered_set<size_t> > proc_recv_labels;
      std::unordered_map< int , std::unordered_set<size_t> > proc_send_labels;
      for(auto& p : label_id_transition)
      {
        // 1) transitive closure
        ssize_t dst_label_id = p.second;
        auto it = label_id_transition.find( dst_label_id );
        while( it != label_id_transition.end() ) { dst_label_id = it->second; it = label_id_transition.find(dst_label_id); }
        p.second = dst_label_id; // transform label transition destination to final label value
        
        // 2) cc table label index map
        if( label_idx_map.find(p.second)==label_idx_map.end() ) label_idx_map.insert( { p.second , label_idx_map.size() } );
        const size_t dst_label_idx = label_idx_map[p.second];
        
        // 3) send/receive labels in cc_table for each MPI partner
        if( p.first>=local_id_start && p.first<local_id_end )
        {
          assert( ! ( p.second>=local_id_start && p.second<local_id_end ) );
          // a label generated locally, is actually owned by a distant MPI process
          // thus, this process will send the owner local contributions to CC info
          // such that the CC owner (and only it) will gather complete CC information
          const int partner = p.second / max_local_labels;
          proc_send_labels[partner].insert( dst_label_idx );
        }
        else if( p.second>=local_id_start && p.second<local_id_end )
        {
          assert( ! ( p.first>=local_id_start && p.first<local_id_end ) );
          // receive contribution from foreign MPI processes to complete locally owned CC information
          const int partner = p.first / max_local_labels;
          proc_recv_labels[partner].insert( dst_label_idx );
        }
      }


      /*******************************************************************
       * add local contributions to cc info table, avoiding ghost layers
       *******************************************************************/
      for( ssize_t k=gl ; k < (grid_dims.k-gl) ; k++)
      for( ssize_t j=gl ; j < (grid_dims.j-gl) ; j++)
      for( ssize_t i=gl ; i < (grid_dims.i-gl) ; i++)
      {        
        const IJK cell_loc = {i,j,k};
        const IJK domain_cell_loc = cell_loc + grid_offset ;
        for( ssize_t sk=0 ; sk<subdiv ; sk++)
        for( ssize_t sj=0 ; sj<subdiv ; sj++)
        for( ssize_t si=0 ; si<subdiv ; si++)
        {
          const IJK subcell_loc = {si,sj,sk};
          const ssize_t cell_index = grid_ijk_to_index( grid_dims , cell_loc );
          const ssize_t subcell_index = grid_ijk_to_index( IJK{subdiv,subdiv,subdiv} , subcell_loc );
          const ssize_t value_index = cell_index * stride + subcell_index;
          assert( value_index >= 0 );
          const double label = cc_label_ptr[ value_index ];
          if( label >= 0.0 )
          {
            ssize_t label_idx = static_cast<ssize_t>( label );
            assert( label_idx_map.find(label_idx) != label_idx_map.end() );
            label_idx = label_idx_map[label_idx];
            assert( label_idx >= 0 && label_idx < cc_table->size() );
            assert( cc_table->at(label_idx).m_label == -1.0 || cc_table->at(label_idx).m_label == label );            
            cc_table->at(label_idx).m_label = label;
            assert( cc_table->at(label_idx).m_local_idx == -1 || cc_table->at(label_idx).m_local_idx == label_idx );
            cc_table->at(label_idx).m_local_idx = label_idx;
            cc_table->at(label_idx).m_cell_count += 1;
            cc_table->at(label_idx).m_center += make_vec3d( ( domain_cell_loc * subdiv ) + subcell_loc );
          }
        }
      }


      /********************************************************************************************
       * Exchange contributions with foreign MPI processes, such that each CC label owner receives
       * necessary local contributions to complete owned CC informations
       ********************************************************************************************/
      std::unordered_map< int , std::vector<ConnectedComponentInfo> > proc_recv_cc_info;
      std::vector<MPI_Request> recv_requests( proc_recv_labels.size() , MPI_REQUEST_NULL );
      size_t recv_count = 0;
      for(const auto & recv : proc_recv_labels)
      {
        const size_t cc_recv_count = recv.second.size();
        proc_recv_cc_info[ recv.first ].assign( cc_recv_count , ConnectedComponentInfo{} );
        MPI_Irecv( proc_recv_cc_info[recv.first].data() , sizeof(ConnectedComponentInfo) * cc_recv_count , MPI_CHAR , recv.first , 0 , *mpi , & recv_requests[recv_count] );
        ++ recv_count;
      }
      assert( recv_count == proc_recv_labels.size() );
      
      std::unordered_map< int , std::vector<ConnectedComponentInfo> > proc_send_cc_info;
      std::vector<MPI_Request> send_requests( proc_send_labels.size() , MPI_REQUEST_NULL );
      size_t send_count = 0;
      for(const auto & send : proc_send_labels)
      {
        const size_t cc_send_count = send.second.size();
        for(const auto & send_label_idx : send.second)
        {
          proc_send_cc_info[ send.first ].push_back( cc_table->at(send_label_idx) );
        }
        assert( proc_send_cc_info[ send.first ].size() == cc_send_count );
        MPI_Isend( proc_send_cc_info[send.first].data() , sizeof(ConnectedComponentInfo) * cc_send_count , MPI_CHAR , send.first , 0 , *mpi , & send_requests[send_count] );
        ++ send_count;
      }
      assert( send_count == proc_send_labels.size() );

      // wait for messages exchange to terminate
      MPI_Waitall( recv_count , recv_requests.data() , MPI_STATUSES_IGNORE );
      MPI_Waitall( send_count , send_requests.data() , MPI_STATUSES_IGNORE );
      
      // merge informations of cc_info from foreign MPI processes
      for(const auto & recv_mesg : proc_recv_cc_info )
      {
        ldbg << "merge CC info from P"<<recv_mesg.first<<std::endl;
        for(const auto & cc_info : recv_mesg.second)
        {
          ldbg << "\tmerge CC #"<<cc_info.m_label<<std::endl;
          assert( cc_info.m_local_idx != -1 ); // it has been filled in
          assert( label_idx_map.find(static_cast<ssize_t>(cc_info.m_label)) != label_idx_map.end() );
          ssize_t cc_table_idx = label_idx_map[ static_cast<ssize_t>(cc_info.m_label) ];
          assert( cc_table_idx >= 0 && cc_table_idx < cc_table->size() );
          assert( cc_table->at(cc_table_idx).m_label == cc_info.m_label );
          cc_table->at(cc_table_idx).m_cell_count += cc_info.m_cell_count;
          cc_table->at(cc_table_idx).m_center += cc_info.m_center;
        }
      }
      
      /*************************************************************************
       * finalize CC information statistics and filter out some of the CCs
       * dependending on optional filtering parameters
       *************************************************************************/
      const ssize_t cc_table_size = cc_table->size();
      unsigned long long owned_label_count = 0;
      for(ssize_t i=0;i<cc_table_size;i++)
      {
        if( cc_table->at(i).m_local_idx >= 0 )
        {
          assert( cc_table->at(i).m_local_idx == i );
          if( cc_table->at(i).m_cell_count >= (*cc_count_threshold) )
          {
            if( cc_table->at(i).m_cell_count > 0 ) cc_table->at(i).m_center /= cc_table->at(i).m_cell_count;
            else cc_table->at(i).m_center = Vec3d{0.,0.,0.};
            cc_table->at(i).m_local_idx = owned_label_count;
            cc_table->at(i).m_global_idx = -1;
            cc_table->at(owned_label_count) = cc_table->at(i);
            ++ owned_label_count;
          }
        }
      }
      cc_table->resize(owned_label_count);
      unsigned long long global_label_idx_start = owned_label_count;
      unsigned long long global_label_count = owned_label_count;
      MPI_Exscan( &owned_label_count , &global_label_idx_start , 1 , MPI_UNSIGNED_LONG_LONG , MPI_SUM , *mpi );
      MPI_Allreduce( MPI_IN_PLACE , &global_label_count , 1 , MPI_UNSIGNED_LONG_LONG , MPI_SUM , *mpi ); 
      for(auto & cc : *cc_table) { cc.m_global_idx = cc.m_local_idx + global_label_idx_start; }
      cc_table->m_global_label_count = global_label_count;
      cc_table->m_local_label_start = global_label_idx_start;

      ldbg << "cc_label : owned_label_count="<< owned_label_count<<", global_label_count="<<global_label_count<<", global_label_idx_start="<<global_label_idx_start << std::endl;
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
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory("cc_label", make_simple_operator< ConnectedComponentLabel > );
  }

}
