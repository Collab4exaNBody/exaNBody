#pragma once

#include <exanb/compute/compute_pair_singlemat.h>
#include <onika/dac/soatl.h>
#include <exanb/core/operator_task.h>

namespace exanb
{

  // ==== Onika parallel task  ====
  // parallel task is enqueued and post-processed with other enqueued tasks
  template<class GridT, class CPBufFactoryT, class OptionalArgsT, class FuncT, class RWFieldSetT
         , class ROFieldSetT=FieldSet<>, class NbhRWFieldSetT=FieldSet<>, class NbhROFieldSetT=FieldSet<>
         , size_t GrainSize=1 , size_t StencilScale=1 >
  static inline void compute_pair_singlemat_onika(
    onika::task::ParallelTaskQueue& ptq,
    GridT* gridp,
    const double rcut,
    const bool enable_ghosts,
    CPBufFactoryT && cpbuf_factory,
    OptionalArgsT && optional,
    const FuncT* funcp,
    RWFieldSetT ,
    ROFieldSetT = ROFieldSetT{},
    NbhRWFieldSetT = NbhRWFieldSetT{},    
    NbhROFieldSetT = NbhROFieldSetT{},
    std::integral_constant<size_t,GrainSize> = std::integral_constant<size_t,GrainSize>{} ,
    std::integral_constant<size_t,StencilScale> = std::integral_constant<size_t,StencilScale>{}
    )
  {
    using namespace onika;
    using center_ro_slices_t = slices_from_field_set_t< AddDefaultFields<ROFieldSetT> >;
    using center_rw_slices_t = slices_from_field_set_t< RWFieldSetT >;
    using nbh_ro_slices_t = slices_from_field_set_t< AddDefaultFields<NbhROFieldSetT> >;
    using nbh_rw_slices_t = slices_from_field_set_t< NbhRWFieldSetT >;
    
    using per_cell_field_set_t = RemoveFields< MergeFieldSet< ROFieldSetT , RWFieldSetT > , DefaultFields >;
    //using CPBuf = typename CPBufFactoryT::ComputePairBuffer ;
    using coord_t = onika::oarray_t<size_t,3>;

    const double rcut2 = rcut * rcut;

    auto cells = gridp->cells();
    int gl = gridp->ghost_layers();
    if( enable_ghosts ) { gl = 0; }
    const IJK dims = gridp->dimension();

    dac::box_span_t<3,GrainSize> span { {size_t(gl),size_t(gl),size_t(gl)} , { size_t(dims.i-2*gl) , size_t(dims.j-2*gl) , size_t(dims.k-2*gl) } };

    auto cells_view = dac::make_array_3d_view( cells , { size_t(dims.i) , size_t(dims.j) , size_t(dims.k) } );
    
    // stencil is read-only for Rx,Ry,Rz at local and neighborhood positions, and is read-write for user selected fields at local position
    using NbhCellStencil = dac::nbh_3d_stencil_t< dac::stencil_element_t< center_ro_slices_t, center_rw_slices_t > , nbh_ro_slices_t , nbh_rw_slices_t , StencilScale >;
    auto cell = dac::make_access_controler( cells_view , NbhCellStencil{} );

    auto nbh_view = dac::make_array_3d_view( optional.nbh.m_nbh_streams , { size_t(dims.i) , size_t(dims.j) , size_t(dims.k) } );
    auto nbh = dac::make_access_controler( nbh_view );

    // std::cout<<"stencil is local = "<< dac::is_local_stencil_v<NbhCellStencil> << "\n";

    // prepare proper lambda execution
    static constexpr typename decltype(optional.nbh)::is_symmetrical_t symmetrical;
    static constexpr bool use_compute_buffer = ComputePairTraits<FuncT>::ComputeBufferCompatible;    
    static constexpr onika::BoolConst<use_compute_buffer> use_cpbuf = {};
    static constexpr DefaultPositionFields pos_fields = {};
    auto * cp_cells = gridp->cells();
    auto cp_cpbuf = cpbuf_factory;
    auto cp_optional = optional;
    auto cp_func = *funcp;
    auto cp_dims = dims; // useless, just emphasizes the fact dims is copy captured

    ptq <<
      onika_parallel_for( span, cell, nbh )
      {
        // copy captured here : optional, cp_cpbuf, cp_dims, cp_cells, cp_func, rcut2
        const IJK loc_a = { ssize_t(item_coord[0]) , ssize_t(item_coord[1]) , ssize_t(item_coord[2]) };
        const size_t cell_a = grid_ijk_to_index( cp_dims , loc_a );
        compute_pair_singlemat_cell(cp_cells,cp_dims,loc_a,cell_a,rcut2,cp_cpbuf,cp_optional,cp_func,cp_optional.nbh.m_chunk_size,symmetrical,per_cell_field_set_t{},pos_fields,use_cpbuf);
      }
      / onika::task::proxy_set_cuda_t< ComputePairTraits<FuncT>::CudaCompatible > {}
      / [cp_dims,cp_cells] ONIKA_HOST_DEVICE_FUNC ( coord_t c ) -> uint64_t { return cp_cells[ grid_ijk_to_index(cp_dims,IJK{ssize_t(c[0]),ssize_t(c[1]),ssize_t(c[2])}) ].size(); }
      
      /*>>
      onika_task()
      {
        std::cout << "compute_pair_singlemat_onika task done" <<std::endl << std::flush;
      }*/
      ;
  }

}

