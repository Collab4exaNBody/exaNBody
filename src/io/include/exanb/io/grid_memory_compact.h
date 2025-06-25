#include <onika/log.h>
#include <onika/memory/allocator.h>
#include <omp.h>

#include <iostream>
#include <fstream>

namespace exanb
{
  template<typename GridT>
    void grid_memory_compact(GridT& grid, bool enable_grid_compact = true, bool force_realloc = false, bool ghost = true)
    {
      if(!enable_grid_compact) return;

      const bool compact_ghosts = ghost;
      const bool relocate = force_realloc;

      auto cells = grid.cells();
      const auto & cell_allocator = grid.cell_allocator();
      const size_t n_cells = grid.number_of_cells();
      ssize_t memory_before = 0;
      ssize_t memory_after = 0;
      ssize_t payload = 0;
      size_t n_realloc = 0;
      size_t n_resize = 0;
      size_t n_empty = 0;
#     pragma omp parallel
      {
        ssize_t loc_memory_before = 0;
        ssize_t loc_memory_after = 0;
        ssize_t loc_payload = 0;
        size_t loc_n_realloc = 0;
        size_t loc_n_resize = 0;
        size_t loc_n_empty = 0;

        size_t tid = omp_get_thread_num();
        size_t nt = omp_get_num_threads();
        for(size_t i=tid;i<n_cells;i+=nt)
        {
          bool compact_cell = true;
          if( ! compact_ghosts ) { compact_cell = ! grid.is_ghost_cell(i); }
          if( compact_cell )
          {
            loc_payload += cells[i].payload_bytes();
            loc_memory_before += cells[i].memory_bytes();
            size_t capacity_before = cells[i].capacity();
            void* storage_ptr_before = cells[i].storage_ptr();
            if( storage_ptr_before==nullptr ) { ++ loc_n_empty; }
            cells[i].shrink_to_fit( cell_allocator , relocate );
            loc_memory_after += cells[i].memory_bytes();

            if( capacity_before != cells[i].capacity() ) { ++ loc_n_resize; }
            if( storage_ptr_before != cells[i].storage_ptr() ) { ++ loc_n_realloc; }
          }
        }

#       pragma omp critical(gmc_stats_update)
        {
          memory_before += loc_memory_before;
          memory_after += loc_memory_after;
          payload += loc_payload;
          n_realloc += loc_n_realloc;
          n_resize += loc_n_resize;
          n_empty += loc_n_empty;
        }

      }
      if( payload > 0 )
      {
        ldbg << "grid_memory_compact: n_realloc="<<n_realloc<<", n_resize="<<n_resize<<", n_empty="<<n_empty<<", n_cells="<<n_cells<<std::endl;
        ldbg << "grid_memory_compact: saved "<< /*ios::mem_size{*/ memory_before-memory_after /*}*/ << ", overhead "<< ((memory_before-payload)*100)/payload<<"% -> "<<((memory_after-payload)*100)/payload<<"%"<<std::endl;
      }
    }
}
