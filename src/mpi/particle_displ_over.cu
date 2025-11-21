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
#include <exanb/core/grid.h>
#include <onika/math/basic_types.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/backup_r.h>
#include <exanb/core/domain.h>

#include <mpi.h>

#include <exanb/compute/reduce_cell_particles.h>
#include <exanb/mpi/particle_displ_over_async_request.h>

namespace exanb
{
  
  template<bool xform_evolved>
  struct ReduceMaxDisplacementFunctor
  {
    const PositionBackupData::CellPositionBackupVector * m_backup_data = nullptr;
    const IJK m_cell_loc_offset = { 0, 0, 0 };
    const Vec3d m_origin = { 0.0 , 0.0 , 0.0 };
    const double m_cell_size = 0.0;
    const double m_threshold_sqr = 0.0;
    const Mat3d xform_tpre;
    const Mat3d xform_tcur;

    ONIKA_HOST_DEVICE_FUNC inline void operator () (unsigned long long int & count_over_dist2 , IJK cell_loc, size_t cell, size_t j, double rx, double ry, double rz , reduce_thread_local_t={} ) const
    {
      const uint32_t* rb = onika::cuda::vector_data( m_backup_data[cell] );
      Vec3d cell_origin = m_origin + ( cell_loc + m_cell_loc_offset ) * m_cell_size;
      
      Vec3d pos_tpre = Vec3d{restore_u32_double(rb[j*3+0],cell_origin.x,m_cell_size), restore_u32_double(rb[j*3+1],cell_origin.y,m_cell_size), restore_u32_double(rb[j*3+2],cell_origin.z,m_cell_size)};
      Vec3d pos_tcur = Vec3d{rx, ry, rz};

      if constexpr (xform_evolved) {
        pos_tpre = xform_tpre * pos_tpre;
        pos_tcur = xform_tcur * pos_tcur;
      }

      if( onika::math::norm2(pos_tcur-pos_tpre) >= m_threshold_sqr )
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
  template<bool xform_evolved> struct ReduceCellParticlesTraits<exanb::ReduceMaxDisplacementFunctor<xform_evolved>>
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
    ADD_SLOT( Domain             , domain    , INPUT );
    ADD_SLOT( PositionBackupData , backup_r  , INPUT );
    ADD_SLOT( double             , threshold , INPUT , 0.0 );
    ADD_SLOT( double             , threshold_lab , INPUT , 0.0 );
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
      const double max_dist_lab = *threshold_lab;
      const Mat3d xform_tpre = backup_r->m_xform;
      const Mat3d xform_tcur = domain->xform();
      bool has_xform_evolved = !onika::math::is_identity( xform_tpre * inverse( xform_tcur ) );
      const double max_dist2 = max_dist * max_dist;
      const double max_dist2_lab = max_dist_lab * max_dist_lab;
      const double threshold_sqr = has_xform_evolved ? max_dist2_lab : max_dist2;
      ldbg << "max dist       = " << max_dist << std::endl;
      ldbg << "max dist_lab   = " << max_dist_lab << std::endl;
      ldbg << "previous XForm = " << backup_r->m_xform << std::endl;
      ldbg << "current XForm  = " << domain->xform() << std::endl;      
      ldbg << "evolved XForm  = " << has_xform_evolved << std::endl;
      
      const double cell_size = grid->cell_size();

      particle_displ_comm->m_comm = *mpi;
      particle_displ_comm->m_request = MPI_REQUEST_NULL;
      particle_displ_comm->m_particles_over = 0;
      particle_displ_comm->m_all_particles_over = 0;
      particle_displ_comm->m_async_request = false;
      particle_displ_comm->m_request_started = false;

      const IJK dims = grid->dimension();
      const size_t n_cells = grid->number_of_cells();
      ldbg << "ParticleDisplacementOver: n_cells = "<<n_cells<<" , dims = "<<dims<< std::endl;
      assert( backup_r->m_data.size() == n_cells );

      if( *async )
      {
        ldbg << "Async particle_displ_over => result set to false" << std::endl;
        particle_displ_comm->m_async_request = true;
        auto user_cb = onika::parallel::ParallelExecutionCallback{ reduction_end_callback , & (*particle_displ_comm) };

        if (has_xform_evolved) {
          ReduceMaxDisplacementFunctor<true> func = { backup_r->m_data.data() , grid->offset() , grid->origin() , cell_size , threshold_sqr, xform_tpre, xform_tcur };
          reduce_cell_particles( *grid , false , func , particle_displ_comm->m_particles_over , reduce_field_set , parallel_execution_context() , user_cb );
        } else {
          ReduceMaxDisplacementFunctor<false> func = { backup_r->m_data.data() , grid->offset() , grid->origin() , cell_size , threshold_sqr, xform_tpre, xform_tcur };
          reduce_cell_particles( *grid , false , func , particle_displ_comm->m_particles_over , reduce_field_set , parallel_execution_context() , user_cb );
        }
        particle_displ_comm->start_mpi_async_request();
        *result = false;
      }
      else
      {    
        ldbg << "Nb part moved over "<< max_dist <<" (local) = " << particle_displ_comm->m_particles_over << std::endl;
        if (has_xform_evolved) {
          ReduceMaxDisplacementFunctor<true> func = { backup_r->m_data.data() , grid->offset() , grid->origin() , cell_size , threshold_sqr, xform_tpre, xform_tcur };
          reduce_cell_particles( *grid , false , func , particle_displ_comm->m_particles_over , reduce_field_set , parallel_execution_context() );
        } else {
          ReduceMaxDisplacementFunctor<false> func = { backup_r->m_data.data() , grid->offset() , grid->origin() , cell_size , threshold_sqr, xform_tpre, xform_tcur };
          reduce_cell_particles( *grid , false , func , particle_displ_comm->m_particles_over , reduce_field_set , parallel_execution_context() );
        }
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
  ONIKA_AUTORUN_INIT(particle_displ_over)
  {
    OperatorNodeFactory::instance()->register_factory( "particle_displ_over", make_grid_variant_operator< ParticleDisplacementOver > );
  }

}

