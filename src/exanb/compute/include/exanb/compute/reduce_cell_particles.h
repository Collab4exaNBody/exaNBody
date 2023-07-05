#pragma once

#include <onika/task/parallel_task_config.h>
#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_error.h>
#include <onika/cuda/device_storage.h>
#include <onika/soatl/field_id.h>

#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/gpu_execution_context.h>

#ifdef XSTAMP_OMP_NUM_THREADS_WORKAROUND
#include <omp.h>
#endif

namespace exanb
{

  struct reduce_thread_local_t {};
  struct reduce_thread_block_t {};
  struct reduce_global_t {};

  // this template is here to know if compute buffer must be built or computation must be ran on the fly
  template<class FuncT> struct ReduceCellParticlesTraits
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool RequiresCellParticleIndex = false;
    static inline constexpr bool CudaCompatible = false;
  };

  template<class CellsT, class FuncT, class ResultT, class... field_ids>
  ONIKA_DEVICE_KERNEL_FUNC
  ONIKA_DEVICE_KERNEL_BOUNDS(ONIKA_CU_MAX_THREADS_PER_BLOCK,ONIKA_CU_MIN_BLOCKS_PER_SM)
  void reduce_cell_particles_gpu_kernel(
    CellsT* cells,
    IJK dims,
    unsigned int gl,
    const FuncT func,
    ResultT* reduced_val,
    GPUKernelExecutionScratch* scratch,
    FieldSet<field_ids...> )
  {
    ResultT local_val = *reduced_val;
  
    // avoid use of compute buffer when possible
    const IJK dimsNoGL = { dims.i-2*gl , dims.j-2*gl , dims.k-2*gl };
    const uint64_t ncells_no_gl = dimsNoGL.i * dimsNoGL.j * dimsNoGL.k;
    
    {
      ONIKA_CU_BLOCK_SHARED unsigned int cell_a_no_gl;
      do
      {
        if( ONIKA_CU_THREAD_IDX == 0 )
        {
          cell_a_no_gl = ONIKA_CU_ATOMIC_ADD( scratch->counters[0] , 1u );
          //printf("processing cell #%d\n",int(cell_a_no_gl));
        }
        ONIKA_CU_BLOCK_SYNC();
        if( cell_a_no_gl < ncells_no_gl )
        {
          const IJK loc_a_no_gl = grid_index_to_ijk( dimsNoGL, cell_a_no_gl );
          const IJK loc_a = { loc_a_no_gl.i+gl , loc_a_no_gl.j+gl , loc_a_no_gl.k+gl };
          const size_t cell_a = grid_ijk_to_index( dims, loc_a );
          const unsigned int n = cells[cell_a].size();
          ONIKA_CU_BLOCK_SIMD_FOR(unsigned int , i , 0 , n )
          {
            if constexpr ( ReduceCellParticlesTraits<FuncT>::RequiresCellParticleIndex )
            {
              func( local_val, loc_a, cell_a , i , cells[cell_a][onika::soatl::FieldId<field_ids>{}][i] ... , reduce_thread_local_t{} );
            }
            if constexpr ( ! ReduceCellParticlesTraits<FuncT>::RequiresCellParticleIndex )
            {
              func( local_val, cells[cell_a][onika::soatl::FieldId<field_ids>{}][i] ... , reduce_thread_local_t{} );
            }
          }
        }
      }
      while( cell_a_no_gl < ncells_no_gl );
    }

    ONIKA_CU_BLOCK_SHARED ResultT team_val;
    if( ONIKA_CU_THREAD_IDX == 0 ) { team_val = local_val; }
    ONIKA_CU_BLOCK_SYNC();

    if( ONIKA_CU_THREAD_IDX != 0 ) 
    {
      func( team_val, local_val , reduce_thread_block_t{} );
    }
    ONIKA_CU_BLOCK_SYNC();

    if( ONIKA_CU_THREAD_IDX == 0 )
    {
      func( *reduced_val, team_val , reduce_global_t{} );      
    }
    ONIKA_CU_BLOCK_SYNC();
  }

  template<class CellsT, class FuncT, class ResultT, class... field_ids>
  inline void reduce_cell_particles_omp_kernel(
    CellsT* cells,
    IJK dims,
    unsigned int gl,
    const FuncT& func,
    ResultT& reduced_val,
    FieldSet<field_ids...> )
  {
    const IJK block_dims = { dims.i-2*gl , dims.j-2*gl , dims.k-2*gl };

    ResultT team_val = reduced_val;

#   ifdef XSTAMP_OMP_NUM_THREADS_WORKAROUND
    omp_set_num_threads( omp_get_max_threads() );
#   endif

#   pragma omp parallel
    {
      ResultT local_val = team_val;
      //_Pragma("omp critical(dbg_mesg)") { if( local_val!=0 ) std::cout << "thread "<<omp_get_thread_num()<<" init local_val="<<local_val<<std::endl; }
    
      GRID_OMP_FOR_BEGIN(block_dims,_,block_cell_a_loc, schedule(dynamic) )
      {
        const IJK cell_a_loc = block_cell_a_loc + gl;
        const size_t cell_a = grid_ijk_to_index( dims , cell_a_loc );
        const size_t n = cells[cell_a].size();
        for(size_t i=0;i<n;i++)
        {
          if constexpr ( ReduceCellParticlesTraits<FuncT>::RequiresCellParticleIndex )
          {
            func( local_val, cell_a_loc, cell_a , i , cells[cell_a][onika::soatl::FieldId<field_ids>{}][i] ... , reduce_thread_local_t{} );
          }
          if constexpr ( ! ReduceCellParticlesTraits<FuncT>::RequiresCellParticleIndex )
          {
            func( local_val, cells[cell_a][onika::soatl::FieldId<field_ids>{}][i] ... , reduce_thread_local_t{} );
          }
        }
      }
      GRID_OMP_FOR_END

      //_Pragma("omp critical(dbg_mesg)") { if( local_val!=0 ) std::cout << "thread "<<omp_get_thread_num()<<" local_val="<<local_val<<std::endl; }

      func( team_val, local_val , reduce_thread_block_t{} );
    }
    
    func( reduced_val, team_val , reduce_global_t{} );

    //_Pragma("omp critical(dbg_mesg)") { if( reduced_val!=0 ) std::cout << "reduced_val="<<reduced_val<<std::endl; }
  }


  // ==== OpenMP parallel for style implementation ====
  // cells are dispatched to threads using a "#pragma omp parallel for" construct
  template<class GridT, class FuncT, class ResultT, class FieldSetT>
  static inline void reduce_cell_particles(
    GridT& grid,
    bool enable_ghosts ,
    const FuncT& func  ,
    ResultT& reduced_val , // initial value is used as a start value for reduction
    FieldSetT cpfields ,
    GPUKernelExecutionContext * exec_ctx = nullptr ,
    GPUStreamCallback* user_cb = nullptr )
  {
    //ResultT reduced_val = init_value;
    
    const IJK dims = grid.dimension();
    const int gl = enable_ghosts ? 0 : grid.ghost_layers() ;
    auto cells = grid.cells();

    if constexpr ( ReduceCellParticlesTraits<FuncT>::CudaCompatible )
    {
      static_assert( sizeof(ResultT) <= GPUKernelExecutionScratch::MAX_RETURN_SIZE , "return value type size exceeds maximum allowed" );
    
      bool allow_cuda_exec = ( exec_ctx != nullptr );
      if( allow_cuda_exec ) allow_cuda_exec = ( exec_ctx->m_cuda_ctx != nullptr );
      if( allow_cuda_exec ) allow_cuda_exec = exec_ctx->m_cuda_ctx->has_devices();
      if( allow_cuda_exec )
      {
        exec_ctx->check_initialize();
        const unsigned int BlockSize = std::min( static_cast<size_t>(ONIKA_CU_MAX_THREADS_PER_BLOCK) , static_cast<size_t>(onika::task::ParallelTaskConfig::gpu_block_size()) );
        const unsigned int GridSize = exec_ctx->m_cuda_ctx->m_devices[0].m_deviceProp.multiProcessorCount * onika::task::ParallelTaskConfig::gpu_sm_mult()
                                    + onika::task::ParallelTaskConfig::gpu_sm_add();

        auto custream = exec_ctx->m_cuda_stream;

        grid.check_cells_are_gpu_addressable();

        exec_ctx->record_start_event();

        exec_ctx->reset_counters();
        exec_ctx->set_return_data( &reduced_val );
        auto * scratch = exec_ctx->m_cuda_scratch.get();

	      ResultT* ptr = (ResultT*)scratch->return_data;
        ONIKA_CU_LAUNCH_KERNEL(GridSize,BlockSize,0,custream, reduce_cell_particles_gpu_kernel, cells, dims, gl, func, ptr, scratch, cpfields );

        exec_ctx->retrieve_return_data( &reduced_val );
        
        if( user_cb != nullptr )
        {
          user_cb->m_exec_ctx = exec_ctx;
          checkCudaErrors( cudaStreamAddCallback(custream,GPUKernelExecutionContext::execution_end_callback,user_cb,0) );
        }
        else
        {
          exec_ctx->wait();
        }
        return;
      }
    }

    // here we failed launching the kernel on the GPU, so we're executing on CPU with OpenMP
    reduce_cell_particles_omp_kernel(cells, dims, gl, func, reduced_val, cpfields);
    //exec_ctx->m_user_callback = nullptr;
    //exec_ctx->m_user_data = nullptr;
    if( user_cb != nullptr )
    {
      user_cb->m_exec_ctx = exec_ctx;
      ( * user_cb->m_user_callback )( exec_ctx , user_cb->m_user_data );
    }
    else
    {
      exec_ctx->wait(); // shall be effectless
    }
  }

}

