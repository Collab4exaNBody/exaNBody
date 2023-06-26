#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/basic_types.h>

#include <cmath>
#include <mpi.h>

#include <exanb/mpi/particle_displ_over_async_request.h>

namespace exanb
{
  

  struct ParticleDisplOverAsyncEnd : public OperatorNode
  {
    // -----------------------------------------------
    // -----------------------------------------------
    ADD_SLOT( ParticleDisplOverAsyncRequest , particle_displ_comm , INPUT_OUTPUT );
    ADD_SLOT( bool               , result    , OUTPUT );

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
      MPI_Status status;
      MPI_Wait( &(particle_displ_comm->m_request), &status );
      ldbg << "particles over threshold ="<< particle_displ_comm->m_particles_over <<" / "<< particle_displ_comm->m_all_particles_over << std::endl;
      *result = ( particle_displ_comm->m_all_particles_over > 0 );
    }

  };
    
 // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "particle_displ_over_async_end", make_simple_operator< ParticleDisplOverAsyncEnd > );
  }

}

