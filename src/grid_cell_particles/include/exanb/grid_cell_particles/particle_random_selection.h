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
#pragma once

#include <onika/cuda/cuda.h>
#include <cstdint>
#include <cmath>
#include <mpi.h>

namespace exanb
{

  // ---------------------------------------------------------------------
  // splitmix64 : small, fast, well distributed 64 bit integer hash.
  // Used to turn (particle_id , seed) into a reproducible pseudo-random
  // value, independent of particle ordering, MPI rank count or thread
  // scheduling. Same id + seed always yields the same key, on CPU or GPU.
  // ---------------------------------------------------------------------
  ONIKA_HOST_DEVICE_FUNC inline uint64_t splitmix64( uint64_t x )
  {
    x += 0x9E3779B97F4A7C15ULL;
    uint64_t z = x;
    z = ( z ^ ( z >> 30 ) ) * 0xBF58476D1CE4E5B9ULL;
    z = ( z ^ ( z >> 27 ) ) * 0x94D049BB133111EBULL;
    return z ^ ( z >> 31 );
  }

  // deterministic pseudo-random key in [0,1) for a given particle id and seed
  ONIKA_HOST_DEVICE_FUNC inline double particle_random_key( uint64_t id, uint64_t seed )
  {
    const uint64_t h = splitmix64( id ^ splitmix64( seed ) );
    // keep the 53 most significant bits -> uniform double in [0,1)
    return double( h >> 11 ) * ( 1.0 / 9007199254740992.0 /* 2^53 */ );
  }

  // ---------------------------------------------------------------------
  // Describes how many particles (among those passing some spatial /
  // logical filter) must be selected :
  //  - ALL      : every eligible particle is selected (plain region selection)
  //  - FRACTION : each eligible particle is selected independently with
  //               probability = fraction. Cheap (single pass, no extra
  //               communication), but the resulting count is only exact
  //               in expectation.
  //  - COUNT    : exactly (up to hash collisions, i.e. for all practical
  //               purposes exactly) 'count' particles are selected globally,
  //               across all MPI ranks, chosen uniformly at random among
  //               eligible particles.
  // ---------------------------------------------------------------------
  struct ParticleSelectionCount
  {
    enum Mode { ALL = 0, FRACTION = 1, COUNT = 2 };
    Mode mode = ALL;
    double fraction = 1.0;
    uint64_t count = 0;
    uint64_t seed = 0;
  };

  struct ParticleSelectionThreshold
  {
    double tau = 1.0;              // select a particle if particle_random_key(id,seed) < tau
    uint64_t global_eligible = 0;  // total nb of particles passing the spatial/logical filter, all ranks
    uint64_t global_target = 0;    // nb of particles the selection is aiming for, all ranks
  };

  // Computes the selection threshold tau such that, among particles passing
  // the caller's filter, testing (particle_random_key(id,seed) < tau) selects
  // the number of particles requested by 'sel', globally across all ranks.
  //
  // The caller provides two local (rank-only) callbacks :
  //   count_eligible()      -> nb of local particles passing the spatial/logical filter
  //   count_below(tau)      -> nb of local eligible particles whose key < tau
  //
  // For FRACTION and ALL modes this only costs one MPI_Allreduce (to know
  // how many particles are eligible in total). For COUNT mode, an extra
  // bisection search on tau is run (each iteration costs one MPI_Allreduce
  // of a local linear scan over already-filtered particles) until the global
  // selected count matches the target exactly, or double precision is
  // exhausted (~50 iterations, negligible compared to a typical particle count).
  template<class CountEligibleFunc, class CountBelowThresholdFunc>
  inline ParticleSelectionThreshold compute_particle_selection_threshold(
        MPI_Comm comm,
        const ParticleSelectionCount& sel,
        CountEligibleFunc count_eligible,
        CountBelowThresholdFunc count_below )
  {
    ParticleSelectionThreshold result;

    const uint64_t local_eligible = count_eligible();
    uint64_t global_eligible = 0;
    MPI_Allreduce( (void*) &local_eligible, (void*) &global_eligible, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm );
    result.global_eligible = global_eligible;

    if( sel.mode == ParticleSelectionCount::ALL || global_eligible == 0 )
    {
      result.tau = 1.0;
      result.global_target = global_eligible;
      return result;
    }

    uint64_t target = 0;
    if( sel.mode == ParticleSelectionCount::FRACTION )
    {
      const double f = sel.fraction < 0.0 ? 0.0 : ( sel.fraction > 1.0 ? 1.0 : sel.fraction );
      target = static_cast<uint64_t>( std::llround( f * double(global_eligible) ) );
    }
    else // COUNT
    {
      target = sel.count;
    }

    if( target >= global_eligible )
    {
      result.tau = 1.0;
      result.global_target = global_eligible;
      return result;
    }
    if( target == 0 )
    {
      result.tau = 0.0;
      result.global_target = 0;
      return result;
    }
    result.global_target = target;

    if( sel.mode == ParticleSelectionCount::FRACTION )
    {
      // single pass Bernoulli trial : expected count is exact, actual
      // count may fluctuate slightly. No need for the bisection search.
      result.tau = double(target) / double(global_eligible);
      return result;
    }

    // COUNT mode : bisection search for the exact global count.
    double lo = 0.0, hi = 1.0;
    double tau = double(target) / double(global_eligible); // good initial guess
    for( int it=0 ; it<64 ; it++ )
    {
      const uint64_t local_below = count_below( tau );
      uint64_t global_below = 0;
      MPI_Allreduce( (void*) &local_below, (void*) &global_below, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm );
      if( global_below == target ) { lo = hi = tau; break; }
      else if( global_below < target ) { lo = tau; }
      else { hi = tau; }
      const double next_tau = 0.5 * ( lo + hi );
      if( next_tau == tau ) break; // double precision exhausted
      tau = next_tau;
    }
    result.tau = tau;
    return result;
  }

}
