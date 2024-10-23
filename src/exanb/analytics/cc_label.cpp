#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/domain.h>
#include <exanb/grid_cell_particles/grid_cell_values.h>
#include <exanb/core/grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/mpi/grid_update_ghosts.h>

#include <mpi.h>
#include <onika/silent_use.h>

namespace exanb
{
  struct ConnectedComponentInfo
  {
    int64_t m_label = -1;
    uint64_t m_cell_count = 0;
    IJK m_seed_loc = {-1,-1,-1};
    double m_mass = 0.;
    double m_volume = 0.;
    Vec3d m_geom_center = {0.,0.,0.};
    Vec3d m_mass_center = {0.,0.,0.};
    Mat3d m_inertial_tensor = { 0.,0.,0., 0.,0.,0., 0.,0.,0. };
  };

  using ConnectedComponentTable = std::vector< ConnectedComponentInfo >;

  class ConnectedComponentLabel : public OperatorNode
  {
    using UpdateGhostsScratch = typename UpdateGhostsUtils::UpdateGhostsScratch;

    ADD_SLOT( MPI_Comm       , mpi                 , INPUT , MPI_COMM_WORLD , DocString{"MPI communicator"} );
    ADD_SLOT( Domain         , domain              , INPUT , REQUIRED );
    ADD_SLOT( std::string    , grid_cell_field     , INPUT , "density" , DocString{"grid cell value field to act as coonnected component mask"} );
    ADD_SLOT( double         , grid_cell_threshold , INPUT , 1. , DocString{"Treshold to determine wheter a cell is selected or not as part of a connected component"} );
    ADD_SLOT( GridCellValues , grid_cell_values    , INPUT_OUTPUT );
    ADD_SLOT( ConnectedComponentTable , cc_table   , INPUT_OUTPUT );

    ADD_SLOT( GhostCommunicationScheme , ghost_comm_scheme , INPUT_OUTPUT , REQUIRED );
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

      // propagate minimum CC label id from nearby cells
      auto propagate_minimum_label = [&]()
      {
        unsigned long long label_update_count = 0;   
        do
        {
          grid_update_ghosts( ::exanb::ldbg, *mpi, *ghost_comm_scheme, null_grid_ptr, *domain, grid_cell_values.get_pointer(),
                              *ghost_comm_buffers, pecfunc,pesfunc, update_fields,
                              0, false, false, false,
                              true, false , std::integral_constant<bool,false>{} );

          label_update_count = 0;

  #       pragma omp parallel for collapse(3) schedule(static) reduction(+:label_update_count)
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
                      cc_label_ptr[ value_index ] = cc_label_ptr[ nbh_value_index ];
                      ++ label_update_count;
                    }
                  }
                }
              }
            }
          }

          MPI_Allreduce(MPI_IN_PLACE,&label_update_count,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,*mpi);
          
          ldbg << "total label updates = "<<label_update_count<<std::endl;
        } while( label_update_count > 0 );
      };

      // identify unique labels owned by this MPI process
      unsigned long long local_id_start=0, local_id_end=0;
      std::map<double,unsigned long long> local_labels;
      
      auto compact_label_ids = [&](bool first_pass) -> bool
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
              if( first_pass || ( label >= local_id_start && label < local_id_end ) )
              {
                if( local_labels.find(label) == local_labels.end() ) local_labels.insert( { label , local_labels.size() } );
              }
            }
          }
        }
        
        unsigned long long total_label_count = local_labels.size();
        MPI_Exscan( &total_label_count , &local_id_start , 1 , MPI_UNSIGNED_LONG_LONG , MPI_SUM , *mpi );
        local_id_end = local_id_start + total_label_count;
        MPI_Allreduce( MPI_IN_PLACE , &total_label_count , 1 , MPI_UNSIGNED_LONG_LONG , MPI_SUM , *mpi );
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
              const double new_label = static_cast<double>( local_labels[label] );
              if( new_label != label ) labels_updated = true;
              cc_label_ptr[ value_index ] = new_label;
            }
          }
        }
        return labels_updated;
      };
      
      ldbg << "*** PASS 1 ***" << std::endl;
      propagate_minimum_label();
      compact_label_ids( true );
      ldbg << "*** PASS 2 ***" << std::endl;
      propagate_minimum_label();
      compact_label_ids( false );
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
