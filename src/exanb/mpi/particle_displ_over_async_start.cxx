#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/grid.h>
#include <exanb/core/basic_types.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/fields.h>
#include <exanb/core/backup_r.h>
#include <exanb/core/operator_task.h>

#include <onika/dac/soatl.h>
#include <onika/dac/array_view.h>

#include <mpi.h>
#include <exanb/mpi/data_types.h> // data type to MPI enum
#include <exanb/mpi/constants.h> // redefine lambda friendly MPI constants

#include <exanb/mpi/particle_displ_over_async_request.h>

namespace exanb
{
  

  template<typename GridT>
  struct ParticleDisplOverAsyncStart : public OperatorNode
  {
    // -----------------------------------------------
    // -----------------------------------------------
    ADD_SLOT( MPI_Comm           , mpi       , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( GridT              , grid      , INPUT );
    ADD_SLOT( PositionBackupData , backup_r  , INPUT );
    ADD_SLOT( double             , threshold , INPUT );
    ADD_SLOT( ParticleDisplOverAsyncRequest , particle_displ_comm , INPUT_OUTPUT );

    // -----------------------------------------------
    // -----------------------------------------------
    inline std::string documentation() const override final
    {
      return R"EOF(
compute the distance between each particle in grid input and it's backup position in backup_r input.
sets result output to true if at least one particle has moved further than threshold.
)EOF";
    }

    // -----------------------------------------------
    // -----------------------------------------------
    inline void generate_tasks()  override final
    {
//      std::cout<<"ParticleDisplOverAsyncStart backup_r@"<<backup_r.get_pointer()<<std::endl;  
      using namespace onika;
      using ro_slices = dac::DataSlices< dac::field_array_size_t, field::_rx, field::_ry, field::_rz >;
      using rw_slices = dac::DataSlices<>;

      onika_operator_task( grid , backup_r , threshold , particle_displ_comm , mpi )
      {
        // interest for auto here, is to be able to easily switch between single and double precision floats if needed.
        double dist2_threshold = threshold * threshold;
        //lout_stream()<<"threshold="<<threshold<<", dist2_threshold="<<dist2_threshold<<std::endl;
        //double max_dist2 = 0.0;
        
        //size_t n_cells = grid->number_of_cells();
        IJK dims = grid.dimension();
        size_t gl = grid.ghost_layers();
        auto cells = grid.cells();
        const double cell_size = grid.cell_size();
        const Vec3d grid_origin = grid.origin() + grid.offset() * cell_size;
        assert( backup_r.m_data.size() == grid.number_of_cells() );

        //assert( gl == 1 );
        
        auto cells_view = dac::make_array_3d_view( cells , { size_t(dims.i) , size_t(dims.j) , size_t(dims.k) } );
        dac::local_stencil_t< ro_slices , rw_slices > cell_stencil{};
        auto cell = dac::make_access_controler( cells_view , cell_stencil );

        auto backup_view = dac::make_array_3d_view( backup_r.m_data.data() , { size_t(dims.i) , size_t(dims.j) , size_t(dims.k) } );
        auto backup = dac::make_access_controler( backup_view , make_default_ro_stencil(backup_view) );
  
        // we span interior (non ghost) cells only      
        dac::box_span_t<3> span { {gl,gl,gl} , {size_t(dims.i-2*gl) , size_t(dims.j-2*gl) , size_t(dims.k-2*gl)} };

        particle_displ_comm.m_particles_over = 0;
        auto over_count = dac::make_reduction_access_controler( particle_displ_comm.m_particles_over , dac::reduction_add );

        ptask_queue() << 
        onika_parallel_for( span , cell , backup , over_count )
        {
          auto && [n_particles, rx,ry,rz] = cell;
          //onika_bind_vars(cell, n_particles, rx,ry,rz );
        
          IJK loc = { ssize_t(item_coord[0]) , ssize_t(item_coord[1]) , ssize_t(item_coord[2]) };
          const Vec3d cell_origin = grid_origin + loc * cell_size;

          static_assert( dac::is_const_lvalue_ref_v<decltype(backup)> , "expected a const reference here" );
          assert( backup.size() == n_particles*3 );
          const uint32_t* rb = backup.data();

          unsigned long n_over = 0;
//#         pragma omp simd reduction(+:n_over)
          for(size_t j=0;j<n_particles;j++)
          {
            double dx = restore_u32_double(rb[j*3+0],cell_origin.x,cell_size) - rx[j];
            double dy = restore_u32_double(rb[j*3+1],cell_origin.y,cell_size) - ry[j];
            double dz = restore_u32_double(rb[j*3+2],cell_origin.z,cell_size) - rz[j];
            double d2 = dx*dx + dy*dy + dz*dz;
            n_over += ( d2 >= dist2_threshold );
          }
          over_count += n_over;
        }
        
        >> 
        
        onika_task( mpi , particle_displ_comm )
        {
          particle_displ_comm.m_request = MPI_REQUEST_NULL;
          // lout_stream() << "start: particle over threshold =" << particle_displ_comm.m_particles_over << std::endl;
          MPI_Iallreduce(
            (const void*) & particle_displ_comm.m_particles_over ,
            (void*) & particle_displ_comm.m_all_particles_over ,
            1,
            mpi_typeof(particle_displ_comm.m_particles_over) ,
            MPI_MAX,
            mpi,
            & particle_displ_comm.m_request );
        }
        
        >>
        
        task::flush() ;
        
      };

    }

  };
    
 // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "particle_displ_over_async_start", make_grid_variant_operator< ParticleDisplOverAsyncStart > );
  }

}

