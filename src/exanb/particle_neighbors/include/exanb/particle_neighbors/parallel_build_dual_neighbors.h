#include <exanb/particle_neighbors/grid_particle_neighbors.h>

namespace exanb
{

  // build complementary bonds list. means that for each particle Pa, it stores bonds to Pb such that Pb is "before" Pa
  void parallel_build_dual_neighbors(const GridParticleNeighbors& primary, GridParticleNeighbors& dual);
  bool check_dual_neighbors(GridParticleNeighbors& dual);

} // namespace exanb

