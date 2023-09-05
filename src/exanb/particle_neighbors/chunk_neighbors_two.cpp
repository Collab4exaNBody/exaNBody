#pragma xstamp_cuda_enable

#pragma xstamp_grid_variant

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/amr/amr_grid.h>
#include <exanb/core/grid.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/amr/amr_grid_algorithm.h>
#include <exanb/core/particle_type_pair.h>
#include <exanb/core/profiling_tools.h>

#include <onika/memory/allocator.h>
#include <onika/cuda/cuda_context.h>
#include <onika/cuda/cuda_math.h>
#include <onika/cuda/cuda.h>
#include <onika/fixed_capacity_vector.h>
#include <onika/memory/memory_partition.h>
#include <onika/cuda/device_storage.h>
#include <onika/cuda/memcpy.h>

#include <exanb/particle_neighbors/chunk_neighbors.h>

#include <exanb/particle_neighbors/chunk_neighbors_config.h>
#include <exanb/particle_neighbors/chunk_neighbors_scratch.h>
#include <exanb/particle_neighbors/chunk_neighbors_fixed_capacity_temp.h>
#include <exanb/particle_neighbors/chunk_neighbors_variable_capacity_temp.h>
#include <exanb/particle_neighbors/chunk_neighbors_encode_cell_stream.h>

#ifdef XSTAMP_CUDA_VERSION
#include <exanb/particle_neighbors/chunk_neighbors_gpu_kernel.h>
#endif

namespace exanb
{
  

  template<typename GridT>
  struct ChunkNeighborsLightWeight : public OperatorNode
  {
    ADD_SLOT( GridT               , grid          , INPUT );
    ADD_SLOT( AmrGrid             , amr           , INPUT );
    ADD_SLOT( double              , nbh_dist      , INPUT );  // value added to the search distance to update neighbor list less frequently
    ADD_SLOT( GridChunkNeighbors  , chunk_neighbors , INPUT_OUTPUT );

    ADD_SLOT( ChunkNeighborsConfig, config , INPUT, ChunkNeighborsConfig{} );
    ADD_SLOT( bool, enable_cuda , INPUT, false );

    ADD_SLOT( ChunkNeighborsScratchEncoding, chunk_neighbors_scratch, PRIVATE );

    inline void execute () override final
    {      
      unsigned int cs = config->chunk_size;
      unsigned int cs_log2 = 0;
      while( cs > 1 )
      {
        assert( (cs&1)==0 );
        cs = cs >> 1;
        ++ cs_log2;
      }
      cs = 1<<cs_log2;
      ldbg << "cs="<<cs<<", log2(cs)="<<cs_log2<<std::endl;
      if( cs != static_cast<size_t>(config->chunk_size) )
      {
        lerr<<"chunk_size is not a power of two"<<std::endl;
        std::abort();
      }
      
      //using PointerTuple = onika::soatl::FieldPointerTuple< GridT::CellParticles::Alignment , GridT::CellParticles::ChunkSize , field::_rx, field::_ry, field::_rz >;

      if( static_cast<size_t>(config->chunk_size) > GRID_CHUNK_NBH_MAX_CHUNK_SIZE )
      {
        lerr << "chunk_size ("<< (config->chunk_size) <<") beyond the limit of "<<GRID_CHUNK_NBH_MAX_CHUNK_SIZE<<std::endl;
        std::abort();
      }

      auto nbh_config = *config;
#     ifdef XSTAMP_CUDA_VERSION
      if( parallel_execution_context()->has_gpu_context() )
      {
        if( !nbh_config.build_particle_offset )
        {
          ldbg << "INFO: force build_particle_offset to true to ensure Cuda compatibility" << std::endl;
          nbh_config.build_particle_offset = true;
        }
      }
#     endif

//      ssize_t gl = grid->ghost_layers();

      const double max_dist = *nbh_dist;
      //const double max_dist2 = max_dist*max_dist;
      
      const size_t n_cells = grid->number_of_cells();
      chunk_neighbors->clear();
      chunk_neighbors->set_number_of_cells( n_cells );
      chunk_neighbors->set_chunk_size( cs );
      chunk_neighbors->realloc_stream_pool( config->stream_prealloc_factor );

      unsigned int max_threads = omp_get_max_threads();
      if( max_threads > chunk_neighbors_scratch->thread.size() )
      {
        chunk_neighbors_scratch->thread.resize( max_threads );
      }
 
      IJK dims = grid->dimension();
      auto cells = grid->cells();
                
      size_t failed_cells = 0;
      //size_t peak_mem = 0;


#     ifdef XSTAMP_CUDA_VERSION
      // detect if GPU execution is possible
      auto exec_ctx = parallel_execution_context();
      bool allow_cuda_exec = *enable_cuda;
      if( allow_cuda_exec ) allow_cuda_exec = ( chunk_neighbors->m_stream_pool_hint > 0 );
      if( allow_cuda_exec ) allow_cuda_exec = ( exec_ctx != nullptr );
      if( allow_cuda_exec ) allow_cuda_exec = ( exec_ctx->m_cuda_ctx != nullptr );
      if( allow_cuda_exec ) allow_cuda_exec = exec_ctx->m_cuda_ctx->has_devices();
      if( allow_cuda_exec )
      {
        exec_ctx->check_initialize();
        const unsigned int BlockSize = 32; //onika::task::ParallelTaskConfig::gpu_block_size();
        const unsigned int BlocksPerSM = 4;
        const unsigned int GridSize = exec_ctx->m_cuda_ctx->m_devices[0].m_deviceProp.multiProcessorCount * BlocksPerSM;

        auto custream = exec_ctx->m_cuda_stream;        
        exec_ctx->gpu_kernel_start();
        auto * scratch = exec_ctx->m_cuda_scratch.get();        
        exec_ctx->reset_counters();
        size_t dev_scratch_mem_size = config->scratch_mem_per_cell * GridSize;
        onika::cuda::CudaDeviceStorage<uint8_t> dev_scratch_mem = onika::cuda::CudaDeviceStorage<uint8_t>::New( exec_ctx->m_cuda_ctx->m_devices[0] , dev_scratch_mem_size );
        
        exec_ctx->set_return_data( & chunk_neighbors->m_fixed_stream_pool );

        ldbg << "memory part @ " << (void*) (scratch->return_data) 
	           << ", base="<<(void*) chunk_neighbors->m_fixed_stream_pool.m_base_ptr
	           << ", cap="<< chunk_neighbors->m_fixed_stream_pool.m_capacity 
	           <<"\n";

        GridChunkNeighborsGPUWriteAccessor chunk_nbh( (onika::memory::MemoryPartionnerMT*) scratch->return_data , *chunk_neighbors );
        
        //using CellsT = std::remove_cv_t< std::remove_reference_t< decltype( cells[0] ) > >;
        //using FuncT = ChunkNeighbors2GPUFunctor<CellsT>;
        //FuncT func = { cells, dims, 0, grid->origin(), grid->cell_size(), grid->offset(), *amr, nbh_config, max_dist, cs, cs_log2, dev_scratch_mem.get(), config->scratch_mem_per_cell, chunk_nbh, scratch, config->subcell_compaction };
        
        ONIKA_CU_LAUNCH_KERNEL(GridSize,BlockSize,0,custream, chunk_neighbors_gpu_kernel, cells,dims,grid->cell_size(),grid->origin(),grid->offset(),*amr,nbh_config,max_dist,cs,cs_log2, dev_scratch_mem.get(),config->scratch_mem_per_cell , chunk_nbh, scratch, config->subcell_compaction );

        exec_ctx->retrieve_return_data( & chunk_neighbors->m_fixed_stream_pool );
        exec_ctx->gpu_kernel_end();
        exec_ctx->wait();
        ldbg<<"kernel executed\n";
      }
      else
#     endif

      {
#     if defined(XSTAMP_CUDA_VERSION) || ! defined(__INTEL_COMPILER)
#       pragma omp parallel
        {        
          unsigned int tid = omp_get_thread_num();
          assert( /* tid>=0 && */ tid<max_threads ); 
          
          chunk_neighbors_scratch->thread[tid].fixed_capacity_scratch.resize( config->scratch_mem_per_cell );
          uint8_t* thread_local_scratch_mem = chunk_neighbors_scratch->thread[tid].fixed_capacity_scratch.data();
                  
          GRID_OMP_FOR_BEGIN(dims,cell_a,loc_a, schedule(dynamic) reduction(+:failed_cells) )
          //GRID_FOR_BEGIN(dims,cell_a,loc_a )
          {
            bool cell_ok = false;
            GridChunkNeighborsHostWriteAccessor chunk_nbh( *chunk_neighbors );
            
            if( config->subcell_compaction )
            {
              ChunkNeighborFixedCapacityTemp<true> tmp( thread_local_scratch_mem, config->scratch_mem_per_cell, cells[cell_a].size() );
              cell_ok = chunk_neighbors_encode_cell_stream(cells,dims,grid->cell_size(),grid->origin(),grid->offset(),*amr,nbh_config,max_dist,cs,cs_log2,cell_a,loc_a, chunk_nbh, tmp );
            }
            else
            {
              ChunkNeighborFixedCapacityTemp<false> tmp( thread_local_scratch_mem, config->scratch_mem_per_cell, cells[cell_a].size() );
              cell_ok = chunk_neighbors_encode_cell_stream(cells,dims,grid->cell_size(),grid->origin(),grid->offset(),*amr,nbh_config,max_dist,cs,cs_log2,cell_a,loc_a, chunk_nbh, tmp );
            }
            
            if( ! cell_ok )
            {
              ChunkNeighborVariableCapacityTemp tmp( chunk_neighbors_scratch->thread[tid] , cells[cell_a].size() );
              cell_ok = chunk_neighbors_encode_cell_stream(cells,dims,grid->cell_size(),grid->origin(),grid->offset(),*amr,nbh_config,max_dist,cs,cs_log2,cell_a,loc_a, chunk_nbh, tmp ); 
              assert( cell_ok );
              ++ failed_cells;
            }
          }
          //GRID_FOR_END
          GRID_OMP_FOR_END
        }
	ldbg << "failed_cells = "<<failed_cells<<std::endl;
#     else
        lerr << "Internal error: [ intel compiler / no cuda ] combination, CPU implementation has been disabled"<<std::endl;
        std::abort();
#     endif
      }

      if( nbh_config.free_scratch_memory )
      {
        chunk_neighbors_scratch->thread.clear();
        chunk_neighbors_scratch->thread.shrink_to_fit();
      }

      chunk_neighbors->update_stream_pool_hint();
      ldbg << "Chunk neighbors alloc'd size = "<<chunk_neighbors->m_stream_pool_hint<<", nb dyn alloc = "<<chunk_neighbors->m_nb_dyn_alloc<<std::endl;

      if( failed_cells > 0 ) ldbg <<failed_cells<<" cells required fallback method\n";
      //lout << "peak mem per cell = " << peak_mem << std::endl;
    }

  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory("chunk_neighbors2", make_grid_variant_operator< ChunkNeighborsLightWeight > );
  }

}

