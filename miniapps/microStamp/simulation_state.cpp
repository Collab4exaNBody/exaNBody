#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/core/log.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/fields.h>
#include <exanb/core/basic_types.h>

#include <mpi.h>
#include <cstring>

namespace microStamp
{
  using namespace exanb;
  using SimulationState = std::vector<double>;

  template<
    class GridT ,
    class = AssertGridHasFields< GridT, field::_vx, field::_vy, field::_vz >
    >
  struct ComputeSimulationState : public OperatorNode
  {
    ADD_SLOT( MPI_Comm           , mpi                 , INPUT , MPI_COMM_WORLD);
    ADD_SLOT( GridT              , grid                , INPUT , REQUIRED);
    ADD_SLOT( Domain             , domain              , INPUT , REQUIRED);
    ADD_SLOT( SimulationState , simulation_state , OUTPUT );

    inline void execute () override final
    {
      MPI_Comm comm = *mpi;
      GridT& grid = *(this->grid);
      auto & sim_info = *simulation_state;

      auto cells = grid.cells();
      IJK dims = grid.dimension();
      size_t ghost_layers = grid.ghost_layers();
      IJK dims_no_ghost = dims - (2*ghost_layers);

      double kinetic_energy = 0.0;  // constructs itself with 0s
      double potential_energy = 0.;
      size_t total_particles = 0;
      
#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN(dims_no_ghost,_,loc_no_ghosts, reduction(+:potential_energy,kinetic_energy,total_particles) )
        {
          IJK loc = loc_no_ghosts + ghost_layers;
          size_t cell_i = grid_ijk_to_index(dims,loc);

          const double* __restrict__ vx = cells[cell_i][field::vx];
          const double* __restrict__ vy = cells[cell_i][field::vy];
          const double* __restrict__ vz = cells[cell_i][field::vz];
          const double* __restrict__ ep = cells[cell_i].field_pointer_or_null(field::ep);

          double local_kinetic_ernergy = 0.;
          double local_potential_energy = 0.;
          size_t n = cells[cell_i].size();

#         pragma omp simd reduction(+:local_potential_energy,local_kinetic_ernergy)
          for(size_t j=0;j<n;j++)
          {
            const double mass = 1.0;
            Vec3d v { vx[j], vy[j], vz[j] };
            local_kinetic_ernergy += dot(v,v) * mass;
            local_potential_energy += ep[j];
          }
          potential_energy += local_potential_energy;
          kinetic_energy += local_kinetic_ernergy;
          total_particles += n;
        }
        GRID_OMP_FOR_END
      }

      sim_info.resize(3,0.0);
      sim_info[0] = kinetic_energy;
      sim_info[1] = potential_energy;
      sim_info[2] = total_particles;
      MPI_Allreduce(MPI_IN_PLACE,sim_info.data(),3,MPI_DOUBLE,MPI_SUM,comm);
    }
  };
    
  template<class GridT> using ComputeSimulationStateTmpl = ComputeSimulationState<GridT>;
    
  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "simulation_state", make_grid_variant_operator< ComputeSimulationStateTmpl > );
  }

}

