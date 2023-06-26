#pragma once

#include <exanb/core/domain.h>
#include <exanb/core/check_particles_inside_cell.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/log.h>

#include <vector>

namespace exanb
{
  

  struct NullParticleOptionalBuffer;
  
  struct NullCellParticleOptionalBuffer
  {
    static inline constexpr void copy_incoming_particle( NullParticleOptionalBuffer&, size_t, size_t ) {}
  };

  struct NullParticleOptionalBuffer
  {
    NullCellParticleOptionalBuffer m_null_buffer;
    static inline constexpr void initialize( size_t ) {}
    inline NullCellParticleOptionalBuffer* otb_buffer() { return &m_null_buffer; }
    inline NullCellParticleOptionalBuffer* cell_buffer( size_t ) { return &m_null_buffer; }
    static inline constexpr void pack_cell_particles( size_t, const std::vector<int32_t> &, bool removed_particles = true ){}
  };

  template<class GridT>
  struct MoveParticlesHelper
  {
    using ParticleT = typename GridT::CellParticles::TupleValueType;
    using ParticleVector = std::vector<ParticleT>;
    
    struct MovePaticlesScratch
    {
      std::vector< ParticleVector > m_buffer;
      std::vector< std::vector<ssize_t> > m_removed_particles;
    };
  };

  template<class LDBG, class GridT, class ParticleVector, class MovePaticlesScratch, class ParticleOptionalBuffer>
  inline void move_particles_across_cells( LDBG& ldbg, const Domain& domain, GridT& grid, ParticleVector& otb_particles, MovePaticlesScratch& move_particles_scratch, ParticleOptionalBuffer& opt_buffer )
  {
    using ParticleT = typename GridT::CellParticles::TupleValueType;  
    const auto & cell_allocator = grid.cell_allocator();
    auto cells = grid.cells();
    size_t n_cells = grid.number_of_cells();
    IJK dims = grid.dimension();
    size_t ghost_layers = grid.ghost_layers();
    IJK dims_no_ghost = dims - 2*ghost_layers;
    IJK grid_offset = grid.offset();

//      assert();
    ldbg << "--- start MovePaticlesNode --- offset=" << grid.offset() << ", origin="<< grid.origin() << ", gl="<<ghost_layers<<", gp="<<grid.number_of_ghost_particles()<<std::endl;

    move_particles_scratch.m_buffer.resize( n_cells );
    move_particles_scratch.m_removed_particles.resize( n_cells );
    for(size_t i=0;i<n_cells;i++)
    {
      move_particles_scratch.m_removed_particles[i].clear();
      move_particles_scratch.m_buffer[i].clear();
    }
    otb_particles.clear();
    
    opt_buffer.initialize( n_cells );

#   pragma omp parallel
    {
      GRID_OMP_FOR_BEGIN(dims_no_ghost,_,src_loc_no_ghost, schedule(dynamic) )
      {
        IJK src_loc = src_loc_no_ghost + ghost_layers;
        ssize_t cell_i = grid_ijk_to_index( dims , src_loc );
        const double* __restrict__ rx = cells[cell_i][field::rx];
        const double* __restrict__ ry = cells[cell_i][field::ry];
        const double* __restrict__ rz = cells[cell_i][field::rz];
        size_t n = cells[cell_i].size();
        for(size_t p_i=0;p_i<n;p_i++)
        {
          Vec3d r{rx[p_i],ry[p_i],rz[p_i]};
          Vec3d ro = r;
          IJK dst_loc = domain_periodic_location( domain , r ) - grid_offset; //grid.locate_cell( r );
          if( src_loc != dst_loc || min_distance2_between( ro, grid.cell_bounds(dst_loc) ) >= grid.epsilon_cell_size2() )
          {
            size_t cell_j = n_cells; 
            ParticleVector* particle_buffer = &otb_particles; // particles moving outside the grid are stored in otb_particles
            auto opt_particle_buffer = opt_buffer.otb_buffer();            
            if( grid.contains(dst_loc) )
            {
              bool dst_is_ghost = inside_grid_shell(dims,0,ghost_layers,dst_loc);
              // if destination is a ghost cell, particle goes to the outside list, to be later sent to its new owner processor
              if( ! dst_is_ghost )
              {
                cell_j = grid_ijk_to_index(dims,dst_loc);
                assert( min_distance2_between( r, grid.cell_bounds(dst_loc) ) < grid.epsilon_cell_size2() );
                particle_buffer = & ( move_particles_scratch.m_buffer[cell_j] );
                opt_particle_buffer = opt_buffer.cell_buffer( cell_j );
              }
            }
            // apply periodic condition correction previously computed in domain_periodic_location
            ParticleT pdata = cells[cell_i][p_i];
            pdata[field::rx] = r.x;
            pdata[field::ry] = r.y;
            pdata[field::rz] = r.z;
#           pragma omp critical(move_particles_to_otb)
            {
              particle_buffer->push_back( pdata );
              opt_particle_buffer->copy_incoming_particle( opt_buffer, cell_i, p_i );
            }
            move_particles_scratch.m_removed_particles[cell_i].push_back(p_i);
          }
        }
      }
      GRID_OMP_FOR_END
    }

    size_t out_max = 0;
    size_t out_min = std::numeric_limits<size_t>::max();
    size_t in_max = 0;
    size_t in_min = std::numeric_limits<size_t>::max();
    size_t outgoing_particles = 0;
    size_t incoming_particles = 0;
    size_t total_particles = 0;
#   ifndef NDEBUG
    const size_t expected_total_particles = grid.number_of_particles() ;
#   endif

#   pragma omp parallel
    {
      //std::vector<bool> particle_flags;
      std::vector<int32_t> particle_pack; // order of particles after particle removal
      GRID_OMP_FOR_BEGIN(dims,cell_i,cell_loc, schedule(dynamic) reduction(+:outgoing_particles,incoming_particles,total_particles) reduction(min:out_min,in_min) reduction(max:out_max,in_max) )
      {
        int32_t n = cells[cell_i].size();
        int32_t n_in = move_particles_scratch.m_buffer[cell_i].size();
        int32_t n_out = move_particles_scratch.m_removed_particles[cell_i].size();
        assert( n_out <= n );

        out_max = std::max( out_max, size_t(n_out) );
        out_min = std::min( out_min, size_t(n_out) );
        in_max = std::max( in_max, size_t(n_in) );
        in_min = std::min( in_min, size_t(n_in) );
        outgoing_particles += n_out;
        incoming_particles += n_in;
        total_particles += n;

        if( n_out > 0 )
        {
          particle_pack.resize( n );
          for(int32_t j=0;j<n;j++) { particle_pack[j] = j; }
          for(int32_t j=0;j<n_out;j++) { particle_pack[ move_particles_scratch.m_removed_particles[cell_i][j] ] = -1; }
          for(int32_t j=n-1;j>=0;j--)
          {
            if( particle_pack[j] == -1 )
            {
              if( j < n-1 )
              {
                assert( particle_pack[n-1] != -1 );
                particle_pack[j] = particle_pack[n-1];
              }
              --n;
            }
          }
          for(int32_t j=0;j<n;j++)
          {
            if( j != particle_pack[j] ) cells[cell_i].set_tuple( j , cells[cell_i][ particle_pack[j] ] );
          }
          assert( size_t(n) < particle_pack.size() ); // strictly less than original size, because of "if n_out > 0"
          particle_pack.resize(n);
          cells[cell_i].resize( n , cell_allocator );
        }
        for(int32_t j=0;j<n_in;j++)
        {
          cells[cell_i].push_back( move_particles_scratch.m_buffer[cell_i][j] , cell_allocator );
        }
        // pack particles (overwrite removed ones) AND copy incoming particles
        opt_buffer.pack_cell_particles( cell_i, particle_pack, n_out > 0 );
      }
      GRID_OMP_FOR_END
    }

    assert( total_particles == expected_total_particles );
    ldbg<<"total particles = "<<total_particles<<std::endl;
    ldbg<<"outgoing (total/min/max) = "<<outgoing_particles<<"/"<<out_min<<"/"<<out_max<< std::endl;
    ldbg<<"incoming (total/min/max) = "<<incoming_particles<<"/"<<in_min<<"/"<<in_max<<std::endl;
    ldbg<<"outside particles = "<<otb_particles.size()<<std::endl;
    ldbg<<"--- end move_particles ---"<<std::endl;

    grid.rebuild_particle_offsets();
#   ifndef NDEBUG
    size_t after_total_particles = 0;
    for(size_t i=0;i<n_cells;i++) after_total_particles += cells[i].size();
    assert( after_total_particles == grid.number_of_particles() );    
#   endif
  }


  /*************************************************************************
   * Base operator template to re-order particles that moved outside cells
   *************************************************************************/
  template<class GridT>
  class MovePaticlesAcrossCells : public OperatorNode
  { 
    using ParticleT = typename MoveParticlesHelper<GridT>::ParticleT;
    using ParticleVector = typename MoveParticlesHelper<GridT>::ParticleVector;
    using MovePaticlesScratch = typename MoveParticlesHelper<GridT>::MovePaticlesScratch;

    ADD_SLOT( Domain , domain , INPUT );
    ADD_SLOT( GridT , grid , INPUT_OUTPUT );
    ADD_SLOT( ParticleVector , otb_particles , OUTPUT );
    ADD_SLOT( MovePaticlesScratch, move_particles_scratch , PRIVATE );

  public:
    inline void execute () override final
    {
      NullParticleOptionalBuffer null_opt_buffer = {};
      move_particles_across_cells( ldbg, *domain, *grid, *otb_particles, *move_particles_scratch, null_opt_buffer );
    }    
  };
 
}


