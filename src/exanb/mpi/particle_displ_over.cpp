#pragma xstamp_cuda_enable

#pragma xstamp_grid_variant

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/grid.h>
#include <exanb/core/basic_types.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/fields.h>
#include <exanb/core/backup_r.h>

#include <mpi.h>

#include <exanb/compute/reduce_cell_particles.h>
#include <exanb/mpi/particle_displ_over_async_request.h>

namespace exanb
{
  

  struct ReduceMaxDisplacementFunctor
  {
    const PositionBackupData::CellPositionBackupVector * m_backup_data = nullptr;
    const IJK m_cell_loc_offset = { 0, 0, 0 };
    const Vec3d m_origin = { 0.0 , 0.0 , 0.0 };
    const double m_cell_size = 0.0;
    const double m_threshold_sqr = 0.0;
    
    ONIKA_HOST_DEVICE_FUNC inline void operator () (unsigned long long int & count_over_dist2 , IJK cell_loc, size_t cell, size_t j, double rx, double ry, double rz , reduce_thread_local_t={} ) const
    {
      const uint32_t* rb = onika::cuda::vector_data( m_backup_data[cell] );
      Vec3d cell_origin = m_origin + ( cell_loc + m_cell_loc_offset ) * m_cell_size;
      double dx = restore_u32_double(rb[j*3+0],cell_origin.x,m_cell_size) - rx;
      double dy = restore_u32_double(rb[j*3+1],cell_origin.y,m_cell_size) - ry;
      double dz = restore_u32_double(rb[j*3+2],cell_origin.z,m_cell_size) - rz;
      if( (dx*dx + dy*dy + dz*dz) >= m_threshold_sqr )
      {
        ++ count_over_dist2;
      }
    }

    ONIKA_HOST_DEVICE_FUNC inline void operator () (unsigned long long int & count_over_dist2 , unsigned long long int value, reduce_thread_block_t ) const
    {
      ONIKA_CU_ATOMIC_ADD( count_over_dist2 , value );
    }

    ONIKA_HOST_DEVICE_FUNC inline void operator () (unsigned long long int & count_over_dist2 , unsigned long long int value, reduce_global_t ) const
    {
      ONIKA_CU_ATOMIC_ADD( count_over_dist2 , value );
    }
  };
}

namespace exanb
{
  template<> struct ReduceCellParticlesTraits<exanb::ReduceMaxDisplacementFunctor>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool RequiresCellParticleIndex = true;
    static inline constexpr bool CudaCompatible = true;
  };
}

namespace exanb
{

  template<typename GridT>
  class ParticleDisplacementOver : public OperatorNode
  {
    static constexpr FieldSet<field::_rx,field::_ry,field::_rz> reduce_field_set {};
  
    // -----------------------------------------------
    // -----------------------------------------------
    ADD_SLOT( MPI_Comm           , mpi       , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( GridT              , grid      , INPUT );
    ADD_SLOT( PositionBackupData , backup_r  , INPUT );
    ADD_SLOT( double             , threshold , INPUT , 0.0 );
    ADD_SLOT( bool               , async     , INPUT , false );

    ADD_SLOT( ParticleDisplOverAsyncRequest  , particle_displ_comm , INPUT_OUTPUT );
    ADD_SLOT( bool               , result    , OUTPUT );

  public:
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
    inline void execute ()  override final
    {
      MPI_Comm comm = *mpi;

      // interest for auto here, is to be able to easily switch between single and double precision floats if needed.
      const double max_dist = *threshold;
      const double max_dist2 = max_dist * max_dist;
      const double cell_size = grid->cell_size();

      particle_displ_comm->m_comm = *mpi;
      particle_displ_comm->m_request = MPI_REQUEST_NULL;
      particle_displ_comm->m_particles_over = 0;
      particle_displ_comm->m_all_particles_over = 0;
      particle_displ_comm->m_async_request = false;
      particle_displ_comm->m_request_started = false;

      ReduceMaxDisplacementFunctor func = { backup_r->m_data.data() , grid->offset() , grid->origin() , cell_size , max_dist2 };

      if( *async )
      {
        ldbg << "Async particle_displ_over => result set to false" << std::endl;
        particle_displ_comm->m_async_request = true;
        auto user_cb = onika::parallel::ParallelExecutionCallback{ reduction_end_callback , & (*particle_displ_comm) };
        reduce_cell_particles( *grid , false , func , particle_displ_comm->m_particles_over , reduce_field_set , parallel_execution_context() , user_cb );
        particle_displ_comm->start_mpi_async_request();
        *result = false;
      }
      else
      {    
        reduce_cell_particles( *grid , false , func , particle_displ_comm->m_particles_over , reduce_field_set , parallel_execution_context() );
        MPI_Allreduce( & ( particle_displ_comm->m_particles_over ) , & ( particle_displ_comm->m_all_particles_over ) , 1 , MPI_UNSIGNED_LONG_LONG , MPI_SUM , comm );
        ldbg << "Nb part moved over "<< max_dist <<" (local/all) = "<< particle_displ_comm->m_particles_over <<" / "<< particle_displ_comm->m_all_particles_over << std::endl;
        *result = ( particle_displ_comm->m_all_particles_over > 0 ) ;
      }

    }
    
    static inline void reduction_end_callback( void * userData )
    {
      ::exanb::ldbg << "async CPU/GPU reduction done, start async MPI collective" << std::endl;
      auto * particle_displ_comm = (ParticleDisplOverAsyncRequest*) userData ;
      assert( particle_displ_comm != nullptr );
      assert( particle_displ_comm->m_all_particles_over >= particle_displ_comm->m_particles_over );
      particle_displ_comm->start_mpi_async_request();
    }

  };
    
 // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "particle_displ_over", make_grid_variant_operator< ParticleDisplacementOver > );
  }

}

