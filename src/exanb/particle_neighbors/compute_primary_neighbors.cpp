#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid.h>
#include <exanb/amr/amr_grid.h>
#include <exanb/core/check_particles_inside_cell.h>

#include <exanb/particle_neighbors/check_neighbors.h>
#include <exanb/particle_neighbors/parallel_build_neighbors.h>
#include <exanb/particle_neighbors/grid_particle_neighbors.h>

#include <memory>

namespace exanb
{
  

  template<typename GridT>
  struct ComputePrimaryNeighborsNode : public OperatorNode
  {

      ADD_SLOT( GridT             , grid          , INPUT );
      ADD_SLOT( AmrGrid           , amr           , INPUT );
      ADD_SLOT( double            , nbh_dist      , INPUT );  // value added to the search distance to update neighbor list less frequently
      ADD_SLOT( GridParticleNeighbors , primary_neighbors , INPUT_OUTPUT );

    inline void execute () override final
    {
// gather values attached to slots
      GridT& grid = *(this->grid);
      AmrGrid& amr = *(this->amr);
      GridParticleNeighbors& pb = *primary_neighbors;
      assert( check_particles_inside_cell(grid) );
      
      double neighbors_dist = *nbh_dist;
      ldbg << "ComputePrimaryneighborsNode: dist="<<neighbors_dist<< std::endl;
      
      // call compute method
      parallel_build_neighbors( grid, amr, pb, neighbors_dist );
      
#     ifndef NDEBUG
      assert( check_neighbors( grid, pb ) );
      size_t n_cells = pb.size();
      size_t total_neighbors = 0;
      size_t max_part_nbh = 0;
      for(size_t i=0;i<n_cells;i++)
      {
        total_neighbors += pb[i].neighbors.size();
        size_t n_particles = pb[i].nbh_start.size();
        size_t nbh_start = 0;
        for(size_t j=0;j<n_particles;j++)
        {
          size_t nbh_end = pb[i].nbh_start[j];
          size_t nbh = nbh_end - nbh_start;
          max_part_nbh = std::max( max_part_nbh , nbh );
          nbh_start = nbh_end;
        }
      }
      ldbg<<"primary neighbors count = "<<total_neighbors<<" neighbors_dist="<<neighbors_dist<<", max_part_nbh="<<max_part_nbh<<  std::endl;
#     endif
    }
  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory(
    "compute_primary_neighbors"
    , make_grid_variant_operator< ComputePrimaryNeighborsNode > );
  }

}

