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
    ADD_SLOT( double             , threshold , INPUT );
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

      unsigned long long int n_particles_over_dist = 0;
      ReduceMaxDisplacementFunctor func = { backup_r->m_data.data() , grid->offset() , grid->origin() , cell_size , max_dist2 };
      n_particles_over_dist = reduce_cell_particles( *grid , false , func , n_particles_over_dist, reduce_field_set , gpu_execution_context() );

/*
      ldbg << "a) count over dist "<< max_dist <<" = "<<n_particles_over_dist << std::endl;           
      n_particles_over_dist = 0;
      IJK dims = grid->dimension();
      int gl = grid->ghost_layers();
      auto cells = grid->cells();
      assert( backup_r->m_data.size() == grid->number_of_cells() );      

#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN(dims-2*gl,_,loc, reduction(+:n_particles_over_dist) schedule(dynamic) )
        {
          const size_t i = grid_ijk_to_index( dims, loc+gl );
          const Vec3d cell_origin = grid->cell_position( loc+gl );

          const size_t n_particles = cells[i].size();
          assert( backup_r->m_data[i].size() == n_particles*3 );

          const uint32_t* rb = backup_r->m_data[i].data();
          const auto* __restrict__ rx = cells[i][field::rx];
          const auto* __restrict__ ry = cells[i][field::ry];
          const auto* __restrict__ rz = cells[i][field::rz];

          unsigned long n_over = 0;
#         pragma omp simd reduction(+:n_over)
          for(size_t j=0;j<n_particles;j++)
          {
            double dx = restore_u32_double(rb[j*3+0],cell_origin.x,cell_size) - rx[j];
            double dy = restore_u32_double(rb[j*3+1],cell_origin.y,cell_size) - ry[j];
            double dz = restore_u32_double(rb[j*3+2],cell_origin.z,cell_size) - rz[j];
            if( i==4503 ) { _Pragma("omp critical(dbg_mesg)") { std::cout << "j="<<j<< " : dr="<<dx<<","<<dy<<","<<dz<<std::endl; } }
            size_t is_over = ( ( dx*dx + dy*dy + dz*dz ) >= max_dist2 );
            n_over += is_over;
          }
          n_particles_over_dist += n_over;
        }
        GRID_OMP_FOR_END
      }
      ldbg << "b) count over dist "<< max_dist <<" = "<<n_particles_over_dist << std::endl;
*/

      unsigned long long total_n_over = 0;
      MPI_Allreduce(&n_particles_over_dist,&total_n_over,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,comm);

      ldbg << "Nb part moved over "<< max_dist <<" (local/all) = "<<n_particles_over_dist<<" / "<<total_n_over << std::endl;
      *result = ( total_n_over > 0 ) ;
    }

  };
    
 // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "particle_displ_over", make_grid_variant_operator< ParticleDisplacementOver > );
  }

}

