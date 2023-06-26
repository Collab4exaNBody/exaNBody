#pragma once

#include <exanb/compute/compute_pair_singlemat.h>

namespace exanb
{
  template<typename GridT, typename OptionalArgsT, typename ComputePairBufferFactoryT, typename FuncT, typename FieldSetT , typename PosFieldsT = DefaultPositionFields >
  static inline void compute_pair_singlemat_taskloop(
    GridT& grid,
    double rcut,
    bool enable_ghosts,
    const OptionalArgsT& optional,
    const ComputePairBufferFactoryT& cpbuf_factory,
    const FuncT& func,
    FieldSetT,
    PosFieldsT = PosFieldsT{}
    )
  {
    //using CPBufT = typename ComputePairBufferFactoryT::ComputePairBuffer;

    const double rcut2 = rcut * rcut;
    const IJK grid_dims = grid.dimension();
    int gl = grid.ghost_layers();
    if( enable_ghosts ) { gl = 0; }
    const IJK block_dims = grid_dims - (2*gl);

    //static CPBufT thread_cpbuf[48];

    auto* gridp = &grid;
    auto* optionalp = &optional;
    auto* funcp = &func;
 
#   pragma omp parallel
    {    
#     pragma omp single
      {
        int nt = omp_get_num_threads();
        //std::cout<<"nt="<<nt<<std::endl;
        GRID_OMP_TASKLOOP(block_dims,loc,default(none) firstprivate(block_dims,grid_dims,rcut2,gl,gridp,optionalp,funcp) shared(cpbuf_factory) num_tasks(nt*8) untied mergeable)
        {
          IJK cell_loc = IJK{ loc_i+gl , loc_j+gl , loc_k+gl };
          size_t cell_idx = grid_ijk_to_index( grid_dims , cell_loc );
          compute_pair_singlemat_cell(*gridp,cell_idx,rcut2,cpbuf_factory,*optionalp,*funcp , FieldSetT{} , PosFieldsT{} );
        }
      }
    }
  }


}

