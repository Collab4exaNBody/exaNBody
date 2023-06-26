#pragma once

#include <mpi.h>
#include <cstdint>

namespace exanb
{

  struct ParticleDisplOverAsyncRequest
  {
    MPI_Request m_request;
    size_t m_particles_over = 0;
    size_t m_all_particles_over = 0;
  };

}
