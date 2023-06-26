#pragma xstamp_grid_variant

#undef NDEBUG

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/log.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/check_particles_inside_cell.h>
#include <exanb/grid_cell_particles/grid_cell_values.h>
#include <exanb/core/thread.h>

#include <mpi.h>
#include <vector>
#include <string>
#include <cstring>
#include <list>
#include <algorithm>

#include <exanb/mpi/data_types.h>

// this allows for parallel compilation of templated operator for each available field set


//#define MIGRATE_CELL_DEBUG_PROFILING 1

#ifdef MIGRATE_CELL_DEBUG_PROFILING
#include <exanb/core/profiling_tools.h>
#define FIRST_TIC() ProfilingTimer T0; profiling_timer_start(T0)
#define TIC_TOC_M(mesg) std::cout << mesg <<__LINE__<<" "<<profiling_timer_elapsed_restart(T0)<<std::endl
#else
#define FIRST_TIC() do{}while(0)
#define TIC_TOC_M(mesg) do{}while(0)
#endif

namespace exanb
{
  

  namespace Legacy
  {
    struct CellValueAdd
    {
      template<class T> inline T operator () (const T& a, const T& b) const { return a+b; }
    };
  }

  /*
    Important Hypothesis 1 :
      when cells and their particles are partitionned, all particles are correctly located in their corresponding cell.
  */

  template<class GridT, class CellValueMergeOperatorT=Legacy::CellValueAdd >
  class MigrateCellParticlesLegacy : public OperatorNode
  {
    using CellParticles = typename GridT::CellParticles;
    using ParticleTuple = typename CellParticles::TupleValueType; // decltype( GridT().cells()[0][0] );
    using ParticleBuffer = std::vector<ParticleTuple>;
    using BufListIt = typename std::list<ParticleBuffer>::iterator;
    using GridCellValueType = typename GridCellValues::GridCellValueType;
    using MergeOp = CellValueMergeOperatorT;

    struct alignas(8) GridBlockInfo
    {
      GridBlock m_block;
      bool m_changed = false;
    };

    // fake lock array class, does nothing at all
    struct NullLockArray
    {
      NullLockArray operator [] (size_t) const { return NullLockArray{}; }
      inline void lock() {}
      inline void unlock() {}
    };
    
    // mutex wrapper that uses try_lock combined with a taskyield directive.
    // to be used inside an openmp task only
    template<class MutexT> struct TaskYieldMutexWrapper
    {
      MutexT& m_mutex;
      inline void lock()
      {
        while( ! m_mutex.try_lock() )
        {
#         pragma omp taskyield
        }
      }
      inline void unlock() { m_mutex.unlock(); }
    };

    // mutex array wrapper that returns a TaskYieldMutexWrapper instead of native mutex when accessed through [] operator
    template<class MutexArrayT, class MutexT>
    struct TaskYieldMutexArrayWrapper
    {
      MutexArrayT & m_lock_array;
      inline TaskYieldMutexWrapper<MutexT> operator [] (size_t i) { return TaskYieldMutexWrapper<MutexT>{ m_lock_array[i] }; }
    };

    // -----------------------------------------------
    // Operator slots
    // -----------------------------------------------
    ADD_SLOT( MPI_Comm  , mpi        , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( long      , mpi_tag    , INPUT , 0 );
    ADD_SLOT( MergeOp   , merge_func , INPUT, MergeOp{} );

    ADD_SLOT( GridBlock , lb_block   , INPUT , REQUIRED );
    ADD_SLOT( double    , ghost_dist , INPUT , REQUIRED );
    ADD_SLOT( Domain    , domain     , INPUT , REQUIRED );
    ADD_SLOT( GridT     , grid       , INPUT_OUTPUT);
    
    // optional particles "outside the box", not in the grid. (possibly due to move_particles operator)
    // it is in/out because elements of this array may be swapped (order changed), though it will contain the same element after operation
    ADD_SLOT( ParticleBuffer , otb_particles, INPUT_OUTPUT , DocString{"Particles outside of local processor's grid"} );

    // MPI buffer size for exchanges
    ADD_SLOT( long , buffer_size, INPUT , 100000 , DocString{"Performance tuning parameter. Size of send/receive buffers in number of particles"} );
    
    // threshold, in number of particles, above which a task is generated to copy cell content to send buffer
    ADD_SLOT( long , copy_task_threshold, INPUT , 256 , DocString{"Performance tuning parameter. Number of particles in a cell above which an asynchronous OpenMP task is created to pack particles to send buffer"} );

    // number of receive buffers per peer process. if negative, interpreted as -N * mpi_processes
    ADD_SLOT( long , extra_receive_buffers, INPUT , -1 , DocString{"Performance tuning parameter. Number of extraneous receive buffers allocated allowing for asynchronous (OpenMP task) particle unpacking. A negative value n is interpereted as -n*NbMpiProcs"} );

    // optional : set to true to force complete destruction and creation of the grid even when lb_block hasn't changed
    ADD_SLOT( bool , force_lb_change, INPUT ,false , DocString{"Force particle packing/unpacking to and from send buffers even if a load balancing has not been triggered"} );
    
    // optional per cell set of scalar values
    ADD_SLOT( GridCellValues , grid_cell_values  , INPUT_OUTPUT , OPTIONAL );
        
  public:
    // -----------------------------------------------
    // -----------------------------------------------
    inline std::string documentation() const override final
    {
      return R"EOF(
migrate_cell_particles does 2 things :
1. it repartitions the data accross mpi processes, as described by lb_block.
2. it reserves space for ghost particles, but do not populate ghost cells with particles.
The ghost layer thickness (in number of cells) depends on ghost_dist.
inputs from different mpi process may have overlapping cells (but no duplicate particles).
the result grids (of every mpi processes) never have overlapping cells.
the ghost cells are always empty after this operator.
)EOF";
    }


    // -----------------------------------------------
    // -----------------------------------------------
    inline void execute () override final
    {
      const long cptask_threshold = *copy_task_threshold;
      const int comm_tag = *mpi_tag;
      MPI_Comm comm = *mpi;
      GridBlock out_block = *lb_block;
      size_t comm_buffer_size = *buffer_size;
      bool force_lb_change = *(this->force_lb_change);
      ParticleBuffer& otb_particles = *(this->otb_particles);
      const Domain& domain = *(this->domain);
      GridT& grid = *(this->grid);

      const double max_nbh_dist = *ghost_dist;
      size_t otb_particle_count = otb_particles.size();

      int nprocs = 1;
      int rank = 0;
      MPI_Comm_size(comm,&nprocs);
      MPI_Comm_rank(comm,&rank);

      ldbg << "--- start migrate_cell_particles --- " << std::endl;
      ldbg << "otb_particle_count="<<otb_particle_count << ", max_nbh_dist=" << max_nbh_dist << ", force_lb_change="<< std::boolalpha << force_lb_change ;
      ldbg << ", comm_buffer_size = "<<comm_buffer_size<<" particles, "<<comm_buffer_size*sizeof(ParticleTuple)<<" bytes"<< std::endl;

      FIRST_TIC();
  
      // first: apply domain periodic to otb particles
#     pragma omp parallel for schedule(static)
      for(size_t i=0; i<otb_particle_count; ++i )
      {
        Vec3d r { otb_particles[i][field::rx] , otb_particles[i][field::ry] , otb_particles[i][field::rz] };
        domain_periodic_location( domain, r );
        otb_particles[i][field::rx] = r.x;
        otb_particles[i][field::ry] = r.y;
        otb_particles[i][field::rz] = r.z;
      }
            
      double cell_size = grid.cell_size();
      size_t in_ghost_layers = grid.ghost_layers();
      const size_t out_ghost_layers = std::ceil( max_nbh_dist / cell_size );
      Vec3d origin = grid.origin();

      GridBlock out_block_ghosts = enlarge_block( out_block , out_ghost_layers );

      // only own cells will be used during exchanges
      GridBlock in_block_with_ghosts = grid.block();
      IJK in_block_ghosts_dims = dimension( in_block_with_ghosts );
      assert( in_block_ghosts_dims == grid.dimension() );
      size_t n_in_block_with_ghost_cells = grid_cell_count( in_block_ghosts_dims );
      assert( n_in_block_with_ghost_cells == grid.number_of_cells() );
      GridBlock in_block_no_ghosts = enlarge_block( grid.block() , -in_ghost_layers );
      assert( in_block_with_ghosts.start ==  grid.offset() );
      
#     ifndef NDEBUG
      const double grid_epsilon_cell_size = grid.epsilon_cell_size();
      size_t in_inner_particles = 0;
      size_t in_total_particles = 0;
      GRID_FOR_BEGIN( grid.dimension() , _ , loc )
      {
        bool is_ghost = inside_grid_shell(in_block_ghosts_dims,0,in_ghost_layers,loc);
        size_t n_particles = grid.cell(loc).size();
        if( ! is_ghost ) { in_inner_particles += n_particles; }
        in_total_particles += n_particles;
      }
      GRID_FOR_END
      MPI_Allreduce(MPI_IN_PLACE,&in_inner_particles,1,mpi_datatype<size_t>(),MPI_SUM,comm);
      MPI_Allreduce(MPI_IN_PLACE,&in_total_particles,1,mpi_datatype<size_t>(),MPI_SUM,comm);
      
      AABB in_grid_bounds = grid.grid_bounds();
      AABB out_bounds = enlarge( AABB{ out_block.start*cell_size+origin , out_block.end*cell_size+origin } , max_nbh_dist );
      ldbg << "MigrateCellParticlesNode: max_nbh_dist="<<max_nbh_dist<< ", origin="<<origin<<", block="<<out_block<<", block_ghosts="<<out_block_ghosts<<", bounds="<<out_bounds <<", out_ghost_layers="<<out_ghost_layers << std::endl;
      ldbg << "in_inner_particles="<<in_inner_particles<<", in_total_particles="<<in_total_particles<<std::endl;
      ldbg << "in_ghost_layers="<<in_ghost_layers <<", in_block="<<in_block_no_ghosts << ", in_bounds="<<in_grid_bounds<< std::endl;
#     endif

      // gather all blocks
      GridBlockInfo my_block_info = { out_block, (in_block_no_ghosts!=out_block) || (in_ghost_layers!=out_ghost_layers) || force_lb_change || (max_nbh_dist!=grid.max_neighbor_distance()) };
      static const GridBlockInfo zero_block_info{};
      std::vector<GridBlockInfo> all_out_blocks(nprocs , zero_block_info );
      
      MPI_Allgather( (char*) &my_block_info, sizeof(GridBlockInfo), MPI_CHAR, (char*) all_out_blocks.data(), sizeof(GridBlockInfo), MPI_CHAR, comm);
      assert( all_out_blocks[rank].m_block == my_block_info.m_block);
      assert( all_out_blocks[rank].m_changed == my_block_info.m_changed );
      bool lb_changed = false;
      for(int p=0;p<nprocs;p++)
      {
	      lb_changed = lb_changed || all_out_blocks[p].m_changed;
      }
      ldbg << "LB changed = "<<lb_changed<<std::endl;
      assert( lb_changed || !my_block_info.m_changed );


      // =================================
      // per cell scalar values, if any
      GridCellValueType* cell_scalars = nullptr;
      unsigned int cell_scalar_components = 0;
      if( grid_cell_values.has_value() && lb_changed )
      {
        cell_scalar_components = grid_cell_values->components();
        if( cell_scalar_components > 0 )
        {
          assert( grid_cell_values->data().size() == grid.number_of_cells() * cell_scalar_components );
          cell_scalars = grid_cell_values->data().data();
        }
      }
      if( cell_scalars )
      {
        ldbg << "migrate cell values with "<< cell_scalar_components << " components"<<std::endl;
      }
      // =================================

      // first we count number of times a cell will be used (how many processors' grid it contributes to)
      // cell_use_count as the same size (and covers the same grid block as input grid)
      std::vector<int> cell_use_count( n_in_block_with_ghost_cells, 0 ); // number of processors using each cell
      for(int p=0;p<nprocs;p++)
      {
        GridBlock partner_block = all_out_blocks[p].m_block;
        GridBlock intersect_block = intersection( partner_block , in_block_no_ghosts );      
        if( ! is_empty(intersect_block) )
        {
          IJK intersect_block_dims = dimension(intersect_block);
          GRID_FOR_BEGIN(intersect_block_dims,_,loc)
          {
            IJK abs_loc = loc + intersect_block.start;
            IJK grid_loc = abs_loc - in_block_with_ghosts.start;
            assert( grid.contains(grid_loc) );
            assert( inside_grid_shell( dimension(in_block_with_ghosts),0,in_ghost_layers,grid_loc) == false );
            size_t index = grid_ijk_to_index( in_block_ghosts_dims, grid_loc );
            assert( index>=0 && index<cell_use_count.size() );
            ++ cell_use_count[index];
          }
          GRID_FOR_END
        }
      }

      // we should find no empty cells that are not used by anyone,
      // unless the destination domain does not cover the original one
      //for(size_t i=0;i<n_in_block_with_ghost_cells;i++)
      GRID_FOR_BEGIN(in_block_ghosts_dims,i,cell_loc)
      {
        CellParticles& cell = grid.cell(i);
        //IJK cell_loc = grid_index_to_ijk( in_block_ghosts_dims, i );
        bool is_ghost = inside_grid_shell(in_block_ghosts_dims,0,in_ghost_layers,cell_loc);
	      if( is_ghost )
	      {
	        if( cell_use_count[i]>0 ) { lerr<<"Internal error: cell_use_count["<<i<<"]>0"<<std::endl; std::abort(); }
	      }
        if( cell_use_count[i]==0 && ( is_ghost || lb_changed ) )
        {
	        if( cell.size()>0 && !is_ghost ) { ldbg << "Warning: cell "<<cell_loc<<" with "<<cell.size()<<" particles is lost" << std::endl; }
          cell.clear();
        }
      }
      GRID_FOR_END

      // ldbg << "prepare send buffers" << std::endl;

      // === build send buffers and progressively empty the input grid as cells are copied ===
#     ifndef NDEBUG
      size_t total_particles_to_send = 0;
#     endif
      std::vector<size_t> send_particle_count(nprocs,0);
      std::vector< std::list<ParticleBuffer> > send_buffer( nprocs );
      std::vector< std::vector<GridCellValueType> > cell_value_send_buffer( nprocs );
      size_t send_packet_count = 0;
      
      // feature 1: get particle buffer slice (concurrent access)
      auto cells_ptr = grid.cells();
      auto* __restrict__ cell_use_count_ptr = cell_use_count.data();

#     pragma omp parallel
      {

#       pragma omp single /*nowait*/
        {
          for(int p=0;p<nprocs;p++)
          {
            GridBlock partner_block = all_out_blocks[p].m_block;
            GridBlock intersect_block = intersection( partner_block , in_block_no_ghosts );      
            IJK intersect_block_dims = dimension(intersect_block);
            AABB partner_bounds = { partner_block.start*cell_size+origin , partner_block.end*cell_size+origin };

            size_t cur_send_buffer_offset = 0;

            if( lb_changed && !is_empty(intersect_block) )
            {
              // enumerate inner particles (not in ghost cells)
              GRID_FOR_BEGIN(intersect_block_dims,_,loc)
              {
                IJK abs_loc = loc + intersect_block.start;

                IJK grid_loc = abs_loc - in_block_with_ghosts.start;
                assert( grid.contains(grid_loc) );
                assert( inside_grid_shell( dimension(in_block_with_ghosts),0,in_ghost_layers,grid_loc) == false );
                
                size_t grid_index = grid_ijk_to_index( in_block_ghosts_dims , grid_loc );
                assert( grid_index < cell_use_count.size() );
                assert( cell_use_count_ptr[grid_index] > 0 );
                size_t n_particles = cells_ptr[grid_index].size();

                ParticleTuple* __restrict__ send_buf_ptr = nullptr;
                if( n_particles > 0 )
                {
                  send_buf_ptr = allocate_send_buffer_storage(n_particles, comm_buffer_size, cur_send_buffer_offset, send_buffer[p] );
                }
                send_particle_count[p] += n_particles;

#               ifndef NDEBUG
                GridBlock partner_block_ghosts = enlarge_block( partner_block , out_ghost_layers ); // this is the partner's whole grid block
                IJK partner_loc = abs_loc - partner_block_ghosts.start;
                assert( inside_grid_shell( dimension(partner_block_ghosts),0,out_ghost_layers,partner_loc) == false );
                const auto* __restrict__ rx = cells_ptr[grid_index][field::rx];
                const auto* __restrict__ ry = cells_ptr[grid_index][field::ry];
                const auto* __restrict__ rz = cells_ptr[grid_index][field::rz];
                for(size_t p_i=0;p_i<n_particles;p_i++)
                {
                  assert( is_inside_threshold( partner_bounds , Vec3d{ rx[p_i], ry[p_i], rz[p_i] } , grid_epsilon_cell_size ) );
                }
                total_particles_to_send += n_particles;
#               endif 

#               pragma omp task default(none) firstprivate(grid_index,send_buf_ptr) shared(cells_ptr,cell_use_count_ptr) if(n_particles>=size_t(cptask_threshold))
                {
                  assert( cell_use_count_ptr[grid_index] > 0 );
                  size_t npart = cells_ptr[grid_index].size();
                  for(size_t p_i=0;p_i<npart;p_i++)
                  {
                    send_buf_ptr[p_i] = cells_ptr[grid_index][p_i];
                  }
                  // this is the only case where it is safe to modify count and free data,
                  // because we know there's only one task accessing this cell
                  if( cell_use_count_ptr[grid_index] == 1 )
                  {
                    cell_use_count_ptr[grid_index] = 0;
                    cells_ptr[grid_index].clear();
                    assert( cells_ptr[grid_index].capacity() == 0 );
                  }
                }

                // build up cell values buffer here
                if( cell_scalars != nullptr )
                {
                  // stores i,j,k as double, not the most efficient but much simpler
                  cell_value_send_buffer[p].push_back( static_cast<GridCellValueType>(abs_loc.i) );
                  cell_value_send_buffer[p].push_back( static_cast<GridCellValueType>(abs_loc.j) );
                  cell_value_send_buffer[p].push_back( static_cast<GridCellValueType>(abs_loc.k) );
                  for(unsigned int c=0; c<cell_scalar_components; c++)
                  {
                    cell_value_send_buffer[p].push_back( cell_scalars[grid_index*cell_scalar_components+c] );
                  }
                }

              }
              GRID_FOR_END
              
            } // if block intersects

            //ldbg << "add otb particles" << std::endl;

            // look if some of the otb_particles goes to the current partner's sub domain
            size_t otb_to_send_end = otb_particle_count;
#           ifndef NDEBUG
            size_t n_outside_particles_sent = 0;
#           endif
            for(size_t otb_i=0; otb_i<otb_particle_count; )
            {
              Vec3d r { otb_particles[otb_i][field::rx] , otb_particles[otb_i][field::ry] , otb_particles[otb_i][field::rz] };
              if( is_inside_exclude_upper( partner_bounds , r ) )
              {                
                  ++ send_particle_count[p];
#                 ifndef NDEBUG
                  ++ n_outside_particles_sent;
#                 endif
                  if( otb_i != (otb_particle_count-1) ) { std::swap( otb_particles[otb_i] , otb_particles[otb_particle_count-1] ); }
                  -- otb_particle_count;
              }
              else { ++ otb_i; }
            }
            
            size_t n_otb_to_send = otb_to_send_end - otb_particle_count;
            if( n_otb_to_send > 0 )
            {
              ParticleTuple* __restrict__ send_buf_ptr = allocate_send_buffer_storage(n_otb_to_send, comm_buffer_size, cur_send_buffer_offset, send_buffer[p] );
              for(size_t otb_i=0; otb_i<n_otb_to_send; otb_i++)
              {
                send_buf_ptr[otb_i] = otb_particles[otb_particle_count+otb_i];
              }
            }
            
            // last buffer, if it exists, is marked have the length of payload size (equals cur_send_buffer_offset)
            if( ! send_buffer[p].empty() )
            {
              mark_send_buffer_end( cur_send_buffer_offset , send_buffer[p].back() );
            }
            
            // packets to local processors won't be sent, they'll be processed locally
            if( p != rank ) { send_packet_count += send_buffer[p].size(); }

#           ifndef NDEBUG
            ldbg << "send "<<send_particle_count[p]<<" particles ("<<n_outside_particles_sent<<" otb) to P"<<p<<" using "<<send_buffer[p].size()<<" buffers" <<std::endl;
#           endif
          }

        } // end of single section

        // other threads will process tasks here
#       pragma omp taskwait
        
        // synchronize
#       pragma omp barrier

        // at this point, all cells must have been visited and cleared.
        // all particles are copied in the send buffers
        // note : n_in_block_with_ghost_cells == grid.number_of_cells()
        if( lb_changed )
        {
#         pragma omp for
          for(size_t i=0;i<n_in_block_with_ghost_cells;i++)
	        {
	          if( cell_use_count_ptr[i]>0 )
	          {
	            assert( cell_use_count_ptr[i]>1 ); // because it should then have been set to 0 by copy task
	            cell_use_count_ptr[i] = 0;
	            cells_ptr[i].clear();
	          }
	        }
        }

        // adjust the size of each send buffer to their payload size + 1 (+1 to hold the size of the next send packet in an uint64_t)
#       pragma omp for
        for(int p=0;p<nprocs;p++)
        {
          for(auto& sb : send_buffer[p]) { resize_send_buffer(sb); }
        }

      } // end of parallel section

      ldbg  << "send_packet_count = " << send_packet_count << std::endl << std::flush;
   
#     ifndef NDEBUG
      size_t total_otb_lost = 0;
      size_t total_otb_sent = 0;
      {
        size_t tmp[3] = {total_particles_to_send,otb_particle_count,otb_particles.size()-otb_particle_count};
        MPI_Allreduce(MPI_IN_PLACE,tmp,3,mpi_datatype<size_t>(),MPI_SUM,comm);
        total_particles_to_send = tmp[0];
        total_otb_lost = tmp[1];
        total_otb_sent = tmp[2];
        ldbg << "total to send = " << total_particles_to_send << ", total otb lost in space = " <<total_otb_lost<<", total otb to send = " << total_otb_sent <<std::endl;
      }
#     endif

      TIC_TOC_M( "P"<<rank<<"@" );

      // reconfigure the grid to match destination block
      if( lb_changed )
      {
        grid.set_offset( out_block_ghosts.start );
        grid.set_max_neighbor_distance( max_nbh_dist );
        grid.set_dimension( dimension( out_block_ghosts ) );
      }
      else
      {
        assert( in_block_no_ghosts==out_block && in_ghost_layers==out_ghost_layers );
      }
      
      // re-allocate per cell scalars
      if( cell_scalars != nullptr )
      {
        // grid_cell_values->m_data.assign( grid.number_of_cells() * cell_scalar_components , static_cast<GridCellValueType>(0) );
        grid_cell_values->set_grid_dims( grid.dimension() );
        grid_cell_values->set_ghost_layers( grid.ghost_layers() );
        grid_cell_values->data().assign( grid.number_of_cells() * cell_scalar_components , static_cast<GridCellValueType>(0) );
        cell_scalars = grid_cell_values->data().data();
      }

      // number of send requests equals the number of packets to send
      size_t send_requests = send_packet_count;

      // only one request per MPI process is used for receives, so total number of requests is this
      size_t total_requests = send_requests + nprocs;

      // cannot be predicted from particle count anymore, has to trust send_packet_count
      // following is just informative about send packet structures
#     ifndef NDEBUG
      for(int p=0;p<nprocs;p++) if( p != rank )
      {
        size_t count = 0;  
        for(const auto& buf:send_buffer[p]) { count += buf.size() - 1; }
        assert( count == send_particle_count[p] );
      }
#     endif

      // first step, send_particle_count will hold the size of the first packet sent to each process
      // additionnaly, each packet holds the count of particle in the next packet to be received
      static_assert( sizeof(uint64_t)<=sizeof(ParticleTuple) , "ParticleTuple type must be at least as large as uint64_t to hold next packet particle count" );
      for(int p=0;p<nprocs;p++)
      {
        send_particle_count[p] = 0;
        if( p != rank )
        {
          for(auto it=send_buffer[p].begin(); it!=send_buffer[p].end();)
          {
            auto next_it = it; ++next_it;
            uint64_t* next_packet_count_ptr = reinterpret_cast<uint64_t*>( it->data() + it->size() - 1 );
            if( it == send_buffer[p].begin() )
            {
              send_particle_count[p] = it->size() - 1;
            }
            if( next_it != send_buffer[p].end() )
            {
              *next_packet_count_ptr = next_it->size() - 1;
            }
            else
            {
              *next_packet_count_ptr = 0;
            }
            it = next_it;
          }
        }
      }
      assert( send_particle_count[rank] == 0 );
 
      // remove last element ( supposed to hold next packet particle count ) while these buffers will be processed
      // locally and the size of each buffer will be interpreted as the number of particles in it.
      for(auto& buf:send_buffer[rank])
      {
        buf.resize( buf.size() - 1 );
      }
 
#     ifndef NDEBUG
      for(int p=0;p<nprocs;p++) if( p != rank )
      {
        ldbg << "send_buffer["<<p<<"].size()="<<send_buffer[p].size()<<" : "<<send_particle_count[p]<<" :";
        for(const auto& buf:send_buffer[p]) { ldbg << " "<< buf.size()-1; }
        ldbg << std::endl;
      }
#     endif
      
      std::vector<size_t> recv_particle_count(nprocs,0);
      MPI_Alltoall( send_particle_count.data(), 1, mpi_datatype<size_t>(), recv_particle_count.data(), 1, mpi_datatype<size_t>(), comm );
      assert( recv_particle_count[rank] == 0 );

      // from now on, recv_particle_count[p] is the number of particles in the first packet to be received from processor p
#     ifndef NDEBUG
      for(int p=0;p<nprocs;p++) if(p!=rank)
      {
        ldbg << "recv_particle_count["<<p<<"] = "<< recv_particle_count[p] << std::endl;
      }
#     endif

      size_t total_particles_sent = 0;
      size_t total_particles_received = 0;

      // intialize array of requests (for both sends and receives)
      std::vector< MPI_Request > requests( total_requests , MPI_REQUEST_NULL );

      //ldbg << "start async sends" << std::endl;

      // prepare per cell scalar sends, and initiate both async sends and receives
      size_t send_cell_value_requests = 0;
      size_t recv_cell_value_requests = 0;
      std::vector<size_t> send_cell_value_count;
      std::vector<size_t> recv_cell_value_count;
      std::vector< std::vector<GridCellValueType> > recv_cell_value_buffer;
      if( cell_scalars )
      {
        ldbg << "start cell value async sends / receives" << std::endl;
        send_cell_value_count.assign(nprocs,0);
        recv_cell_value_count.assign(nprocs,0);
        for(int p=0;p<nprocs;p++)
        {
          if( p != rank )
          {
            send_cell_value_count[p] = cell_value_send_buffer[p].size();
            if( send_cell_value_count[p] >0 ) { ++ send_cell_value_requests; }
          }
        }
        MPI_Alltoall( send_cell_value_count.data(), 1, mpi_datatype<size_t>(), recv_cell_value_count.data(), 1, mpi_datatype<size_t>(), comm );
        assert( recv_cell_value_count[rank] == 0 );      
        for(int p=0;p<nprocs;p++)
        {
          if( recv_cell_value_count[p]>0 ) { ++recv_cell_value_requests; }
        }
        requests.resize( total_requests + nprocs*2 , MPI_REQUEST_NULL );

        recv_cell_value_buffer.resize(nprocs);
        int sr = 0;
        int rr = 0;
        for(int p=0;p<nprocs ;p++)
        {
          if( recv_cell_value_count[p]>0 )
          {
            recv_cell_value_buffer[p].resize( recv_cell_value_count[p] );
            MPI_Irecv( recv_cell_value_buffer[p].data() ,recv_cell_value_count[p], mpi_datatype<GridCellValueType>(), p, comm_tag, comm, & requests[total_requests+nprocs+p] );
            ++ rr;
          }
          if( send_cell_value_count[p]>0 )
          {
            MPI_Isend( cell_value_send_buffer[p].data() , send_cell_value_count[p], mpi_datatype<GridCellValueType>(), p, comm_tag, comm, & requests[total_requests+p] );
            ++ sr;
          }
        }
        assert( static_cast<size_t>(sr)==send_cell_value_requests && sr<=nprocs);
        assert( static_cast<size_t>(rr)==recv_cell_value_requests && rr<=nprocs);
        ldbg << "end of cell value async sends / receives" << std::endl;
      }      


      // =========== initiate all async sends =============
      send_packet_count = 0;
      std::map< int , BufListIt > send_buffer_request_map;
      for(int p=0;p<nprocs;p++)
      {
        if( p != rank )
        {
          for( BufListIt it = send_buffer[p].begin(); it!=send_buffer[p].end() ; ++it)
          {
            assert( send_packet_count < send_requests );
            send_buffer_request_map[send_packet_count] = it;
            size_t send_packet_size = it->size(); //std::min( it->size() , comm_buffer_size );
            assert( send_packet_size > 0 );
            // ldbg << "MPI_Isend "<<send_packet_size-1<<"+1 particles to P"<<p<<" using request #"<<send_packet_count <<std::endl;
            MPI_Isend( (char*) it->data() , send_packet_size*sizeof(ParticleTuple), MPI_CHAR, p, comm_tag, comm, & requests[send_packet_count] );
            ++ send_packet_count;
            total_particles_sent += send_packet_size - 1;
          }
        }
      }
      assert( send_packet_count == send_requests );


      //ldbg << "start first async receives" << std::endl;

      // ================= initiate first async receives ================
      size_t extra_rcv_bufs = 0;
      if( (*extra_receive_buffers) < 0 ) extra_rcv_bufs = - nprocs * (*extra_receive_buffers);
      else extra_rcv_bufs = *extra_receive_buffers;
      //lout << "extra_rcv_bufs = " << extra_rcv_bufs << std::endl;
      std::vector<ParticleBuffer> receive_buffer( nprocs + extra_rcv_bufs);
      size_t active_senders = 0;      
      for(int p=0;p<nprocs;p++)
      {
        if( recv_particle_count[p]>0 )
        {
          assert( p != rank );
          ++ active_senders;
          size_t recv_packet_size = recv_particle_count[p] + 1;  //std::min( recv_particle_count[p] , comm_buffer_size );
          receive_buffer[p].resize( recv_packet_size );
          // ldbg << "MPI_Irecv of "<<recv_packet_size-1<<"+1 particles from P"<<p<<" using request #"<<send_requests+p<<std::endl;
          MPI_Irecv( (char*) receive_buffer[p].data() , recv_packet_size*sizeof(ParticleTuple), MPI_CHAR, p, comm_tag, comm, & requests[send_requests+p] );
        }
      }
      // special atomic pointer array to grab extra receive buffers
      std::atomic<ParticleBuffer*> extra_recv_buffer_ptrs[extra_rcv_bufs]; // small array allocated on the stack
      for(size_t i=0;i<extra_rcv_bufs;i++)
      {
        extra_recv_buffer_ptrs[i] = & receive_buffer[nprocs+i];
      }
      size_t extra_recv_buffer_pos_hint = 0;
      //std::vector<std::mutex> cell_locks( grid.number_of_cells() );
      spin_mutex_array cell_locks( grid.number_of_cells() );
      TaskYieldMutexArrayWrapper<spin_mutex_array,spin_mutex> cell_locks_taskyield { cell_locks };

      // ldbg << active_senders << " async receives initiated" << std::endl;
      //ldbg << "copy local particles" << std::endl;


      // =================== local copy of buffers targeted to myself =============================
      // while first communications are flying, we process information targeted to myself
      size_t total_particles_copied = 0;
#     pragma omp parallel
      {
#       pragma omp single nowait
        {
          for( BufListIt it = send_buffer[rank].begin(); it!=send_buffer[rank].end() ; ++it)
          {
            total_particles_copied += it->size();
            // ldbg << "internal copy of "<< it->size() << " particles"<<std::endl;
            auto* buf_ptr = & (*it);
#           pragma omp task default(none) firstprivate(buf_ptr) shared(grid,out_ghost_layers)
            {
              NullLockArray nla;
              insert_particles_to_grid( grid , nla, *buf_ptr, out_ghost_layers );
              buf_ptr->clear();
              buf_ptr->shrink_to_fit();
            }
          }
        }
#       pragma omp taskwait
      }
      send_buffer[rank].clear();


      // copy local cell values
      if( cell_scalars != nullptr )
      {
        // ldbg << "copy local cell values" << std::endl;
        // import cell values here
        assert( send_cell_value_count[rank] == 0 );
        insert_cell_values( grid, cell_value_send_buffer[rank], cell_value_send_buffer[rank].size(), *grid_cell_values );
      }

      // ldbg << "send/receive particles" << std::endl;

      //size_t n_unpacked_buffers_sync = 0;
      //size_t n_unpacked_buffers_async = 0;

#     pragma omp parallel
      {
      
      if( omp_get_thread_num() == 0 )
      {
      
      // ======================= MPI progression loop ===============================
      // loop until all pending requests are completed
      send_packet_count=0;
      while( active_senders>0 || send_packet_count<send_requests || send_cell_value_requests>0 || recv_cell_value_requests>0 )
      {
        int reqidx = MPI_UNDEFINED;
        MPI_Status status;
        MPI_Waitany( requests.size() /*total_requests*/ ,requests.data(),&reqidx,&status);
        
        if( reqidx != MPI_UNDEFINED )
        {
          assert( requests[reqidx] == MPI_REQUEST_NULL );
          //ldbg << "request #"<<reqidx<<" finished"<<std::endl;
          if( reqidx < ssize_t(send_requests) ) // a send has completed
          {
            //ldbg<<"request #"<<reqidx<<" was a send"<<std::endl;
            // free the corresponding send buffer;
            auto send_buffer_it = send_buffer_request_map.find(reqidx);
            assert( send_buffer_it != send_buffer_request_map.end() );
            // ldbg << "free request #"<<reqidx<<"'s send buffer, its size was "<<send_buffer_it->second->size()<<std::endl;
            send_buffer_it->second->clear();
            send_buffer_it->second->shrink_to_fit(); // really free memory
            send_buffer_request_map.erase( send_buffer_it );
            ++ send_packet_count;
          }
          else if( reqidx < static_cast<ssize_t>(send_requests+nprocs) ) // a receive has completed
          {
            int p = reqidx - send_requests; // p = the sender's rank
            // ldbg<<"request #"<<reqidx<<" was a receive from P"<<p<<std::endl;
            assert(p!=rank && p>=0 && p<nprocs);
            int status_count = 0;
            MPI_Get_count(&status,MPI_CHAR,&status_count);
            size_t received_particles = ( status_count / sizeof(ParticleTuple) ) - 1;
            
            assert( status_count % sizeof(ParticleTuple) == 0 );
            assert( receive_buffer[p].size() == (received_particles+1) );
            // ldbg << "received "<<received_particles<<" particles from P"<<p<<" using request #"<<reqidx <<std::endl;
            
            uint64_t* next_particle_count_ptr = reinterpret_cast<uint64_t*>( receive_buffer[p].data() + receive_buffer[p].size() - 1 );
            recv_particle_count[p] = *next_particle_count_ptr;
            // ldbg << "next_particle_count["<<p<<"]="<<recv_particle_count[p]<<std::endl;
            
            /* --- buffer unpack task --- */
            receive_buffer[p].resize(received_particles); // so that we do not add fake particle holding next particle count
            auto extra_buf = pick_receive_buffer( extra_recv_buffer_ptrs, extra_rcv_bufs, extra_recv_buffer_pos_hint );
            if( extra_buf.second!=nullptr ) // extra receive buffer available, unpack buffer asynchrounously
            {
              assert( extra_buf.first>=0 && extra_buf.first<ssize_t(extra_rcv_bufs) );
              assert( extra_recv_buffer_ptrs[extra_buf.first].load() == nullptr );
              extra_recv_buffer_pos_hint = extra_buf.first + 1;
              std::swap( receive_buffer[p] , * extra_buf.second );
              //++ n_unpacked_buffers_async;
#             pragma omp task default(none) firstprivate(extra_buf) shared(grid,out_ghost_layers,cell_locks_taskyield,extra_recv_buffer_ptrs,extra_rcv_bufs)
              {
                insert_particles_to_grid( grid, cell_locks_taskyield, * extra_buf.second, out_ghost_layers );
                give_receive_buffer_back( extra_recv_buffer_ptrs, extra_rcv_bufs, extra_buf );
              }
            }
            else // synchronous receive buffer unpacking
            {
              insert_particles_to_grid( grid, cell_locks_taskyield, receive_buffer[p], out_ghost_layers );
            }
            /* ------------------------- */
            
            if( recv_particle_count[p] > 0 ) // prepare async receive for next packet to receive
            {
              size_t recv_packet_size = recv_particle_count[p] + 1;
              receive_buffer[p].resize( recv_packet_size );
              MPI_Irecv( (char*) receive_buffer[p].data() , recv_packet_size*sizeof(ParticleTuple), MPI_CHAR, p, comm_tag, comm, & requests[send_requests+p] );
            }
            else
            {
              // release memory as soon as possible
              receive_buffer[p].clear();
              receive_buffer[p].shrink_to_fit();
              -- active_senders;
            }
            total_particles_received += received_particles; 
          }
          else // cell_value recv or send completed
          {
            int cell_reqidx = reqidx - total_requests;
            if( cell_reqidx < nprocs )
            {
              // its a cell value send, p is the destination processor
              int p = cell_reqidx;
              // ldbg<<"request #"<<reqidx<<" was a cell value send to P"<<p<<std::endl;
              assert(p!=rank && p>=0 && p<nprocs);
              // ldbg << "sent cell values to P#"<<p<<std::endl;
              
              // release resources
              cell_value_send_buffer[p].clear();
              cell_value_send_buffer[p].shrink_to_fit();
              -- send_cell_value_requests;
            }
            else
            {
              // its a cell value receive, p is the source processor
              int p = cell_reqidx - nprocs;
              // ldbg<<"request #"<<reqidx<<" was a cell value receive from P"<<p<<std::endl;
              assert(p!=rank && p>=0 && p<nprocs);
              int status_count = 0;
              MPI_Get_count(&status,mpi_datatype<GridCellValueType>(),&status_count);
              assert( status_count == static_cast<ssize_t>(recv_cell_value_count[p]) );              
              // ldbg << "receive cell values from P#"<<p<<std::endl;
              
              // import cell values here
              insert_cell_values( grid, recv_cell_value_buffer[p], recv_cell_value_count[p], *grid_cell_values );

              // release resources
              recv_cell_value_buffer[p].clear();
              recv_cell_value_buffer[p].shrink_to_fit();
              -- recv_cell_value_requests;
            }
          }
        }
        
      } // end of mpi request processing while
    
      } // end of master section

#     pragma omp taskwait

      } // end of parallel section
    
      //std::cout <<"P"<<rank<<" unpacked buffers (sync/async) = "<< n_unpacked_buffers_sync << "/"<<n_unpacked_buffers_async<<std::endl;

      TIC_TOC_M( "P"<<rank<<"@" );

      // complete release of cell values exchange buffers
      recv_cell_value_buffer.clear();
      recv_cell_value_buffer.shrink_to_fit();
      cell_value_send_buffer.clear();
      cell_value_send_buffer.shrink_to_fit();

      //ldbg <<"sent="<<total_particles_sent << std::endl;
      //ldbg <<"received="<<total_particles_received<<", copied="<<total_particles_copied<<std::endl;
      //ldbg <<"total="<<(total_particles_received+total_particles_copied)<<std::endl;

      assert( check_particles_inside_cell(grid) );
      grid.rebuild_particle_offsets();

#     ifndef NDEBUG
      assert( active_senders == 0 );
      assert( send_packet_count == send_requests );
      for( auto& req : requests ) { assert( req == MPI_REQUEST_NULL ); }
//      for(size_t i=0;i<total_requests;i++) { assert( requests[i] == MPI_REQUEST_NULL ); }

      MPI_Allreduce(MPI_IN_PLACE,&total_particles_sent,1,mpi_datatype<size_t>(),MPI_SUM,comm);
      MPI_Allreduce(MPI_IN_PLACE,&total_particles_received,1,mpi_datatype<size_t>(),MPI_SUM,comm);
      MPI_Allreduce(MPI_IN_PLACE,&total_particles_copied,1,mpi_datatype<size_t>(),MPI_SUM,comm);

      ldbg << "total_particles_sent="<<total_particles_sent<<", total_particles_received="<<total_particles_received<<", total_particles_copied="<<total_particles_copied<<std::endl;
      assert( total_particles_sent == total_particles_received );
      if( lb_changed )
      {
        assert( total_particles_sent+total_particles_copied == in_inner_particles+total_otb_sent );      
      }
      else
      {
        assert( total_particles_copied == 0 );      
        assert( total_particles_sent == total_otb_sent );
      }

      size_t out_total_particles = 0;
      IJK out_block_ghosts_dims = dimension( out_block_ghosts );
      GRID_FOR_BEGIN( out_block_ghosts_dims , _ , loc )
      {
        bool is_ghost = inside_grid_shell(out_block_ghosts_dims,0,out_ghost_layers,loc);
        size_t n_particles = grid.cell(loc).size();
        if( is_ghost ) { assert( n_particles == 0) ; }
        out_total_particles += n_particles;
      }
      GRID_FOR_END
      MPI_Allreduce(MPI_IN_PLACE,&out_total_particles,1,mpi_datatype<size_t>(),MPI_SUM,comm);
      assert( out_total_particles == in_inner_particles+total_otb_sent );
#     endif

      ldbg << "--- end migrate_cell_particles ---" << std::endl;
    }
    
    template<class LockArrayT>
    inline void insert_particles_to_grid(GridT& grid, LockArrayT& cell_locks, const std::vector<ParticleTuple>& particles, ssize_t ghost_layers )
    {
      size_t cur_locked_cell = std::numeric_limits<size_t>::max();
      ParticleTuple const * data = particles.data();
      size_t n = particles.size();
      for(size_t i=0;i<n;i++)
      {
        double rx = data[i][field::rx];
        double ry = data[i][field::ry];
        double rz = data[i][field::rz];
        IJK loc = grid.locate_cell( Vec3d{rx,ry,rz} );
        assert( grid.contains(loc) );
        assert( ! inside_grid_shell(grid.dimension(),0,ghost_layers,loc) );
        size_t cell_index = grid.cell_index(loc);
        if( cell_index != cur_locked_cell )
        {
          if( cur_locked_cell != std::numeric_limits<size_t>::max() )
          {
            cell_locks[cur_locked_cell].unlock();
          }
          cell_locks[cell_index].lock();
          cur_locked_cell = cell_index;
        }
        grid.cell(cell_index).push_back( data[i] , grid.cell_allocator() );
      }
      if( cur_locked_cell != std::numeric_limits<size_t>::max() )
      {
        cell_locks[cur_locked_cell].unlock();
      }
    }


     // import values of cell_scalars
    inline void insert_cell_values(GridT& grid, const std::vector<GridCellValueType>& recv_cell_value_buffer, size_t nvalues, GridCellValues& grid_cell_values )
    {
      assert( nvalues == recv_cell_value_buffer.size() );

      size_t cell_scalar_components = grid_cell_values.components();
      GridCellValueType* cell_scalars = grid_cell_values.data().data();
      IJK out_grid_dims = grid.dimension();
            
      //size_t nvalues = recv_cell_value_count[p];
      size_t stride = cell_scalar_components + 3;
      assert( nvalues % stride == 0 );
      nvalues /= stride;
      for(size_t v=0;v<nvalues;v++)
      {
        IJK abs_loc { static_cast<ssize_t>(recv_cell_value_buffer[v*stride+0])
                    , static_cast<ssize_t>(recv_cell_value_buffer[v*stride+1])
                    , static_cast<ssize_t>(recv_cell_value_buffer[v*stride+2]) };
        IJK grid_loc = abs_loc - grid.offset();
        assert( grid.contains(grid_loc) );
        size_t cell_i = grid_ijk_to_index( out_grid_dims , grid_loc );
        for(size_t c=0;c<cell_scalar_components;c++)
        {
          size_t cell_scalars_index = cell_i * cell_scalar_components + c ;
          assert( cell_scalars_index>=0 && cell_scalars_index < grid_cell_values.data().size() );
          assert( v*stride+3+c < recv_cell_value_buffer.size() );
          
          // cell value merge is done through the + operator
          cell_scalars[cell_scalars_index] = (*merge_func)( cell_scalars[cell_scalars_index] , recv_cell_value_buffer[v*stride+3+c] );
        }
      }
    }


    // ====================== send buffer management ========================
    static inline void mark_send_buffer_end(size_t buffer_offset, ParticleBuffer& send_buffer)
    {
      static_assert( sizeof(ParticleTuple) >= sizeof(uint64_t) , "ParticleTuple type must be at least as large as uint64_t to hold next packet particle count" );
      assert( (buffer_offset+1) <= send_buffer.size() );
      uint64_t* buffer_length_ptr = reinterpret_cast<uint64_t*>( send_buffer.data()+send_buffer.size()-1 );
      *buffer_length_ptr = buffer_offset;
    }

    static inline void resize_send_buffer(ParticleBuffer& send_buffer)
    {
      static_assert( sizeof(ParticleTuple) >= sizeof(uint64_t) , "ParticleTuple type must be at least as large as uint64_t to hold next packet particle count" );
      assert( send_buffer.size() >= 1 );
      const uint64_t* buffer_length_ptr = reinterpret_cast<const uint64_t*>( send_buffer.data()+send_buffer.size()-1 );
      const uint64_t buffer_length = *buffer_length_ptr;
      assert( buffer_length <= (send_buffer.size()-1) );
      send_buffer.resize( buffer_length + 1 );
    }

    static inline ParticleTuple* allocate_send_buffer_storage(size_t n_particles, size_t max_buffer_size, size_t& buffer_offset, std::list<ParticleBuffer>& send_buffers)
    {
      assert( n_particles > 0 );
      ParticleBuffer* cur_send_buffer = nullptr;
      
      if( send_buffers.empty() )
      {
        // create a new buffer 
        size_t bufsize = std::max( max_buffer_size , n_particles ) + 1;
        cur_send_buffer = & send_buffers.emplace_back( bufsize );
        buffer_offset = 0;
      }
      else if( ( buffer_offset + n_particles ) > ( send_buffers.back().size() - 1) )
      {
        // shrink previous buffer to avoid empty space in buffer (no deallocation though)
        size_t bufsize = std::max( max_buffer_size , n_particles ) + 1;
        
        // resize last used buffer to fit payload
        mark_send_buffer_end( buffer_offset , send_buffers.back() );
        //send_buffers.back().resize(buffer_offset + 1 );
        
        // create a new buffer
        cur_send_buffer = & send_buffers.emplace_back( bufsize );
        buffer_offset = 0;        
      }
      else
      {
        cur_send_buffer = & send_buffers.back();
      }

      ParticleTuple* buf_ptr = cur_send_buffer->data() + buffer_offset;
      buffer_offset += n_particles;
      return buf_ptr;
    }
    // =====================================================================


    // ======= lock free extra reveive buffer management ===================
    static inline std::pair<ssize_t,ParticleBuffer*> pick_receive_buffer( std::atomic<ParticleBuffer*> * rcv_buffer_ptrs , size_t n_buffers , size_t hint=0 )
    {
      for( size_t j=0; j<n_buffers; ++j )
      {
        size_t i = ( j + hint ) % n_buffers;
        ParticleBuffer* bufptr = rcv_buffer_ptrs[i].exchange( nullptr );
        if( bufptr!=nullptr ) return { i , bufptr };
      }
      return { -1 , nullptr };
    }

    static inline void give_receive_buffer_back( std::atomic<ParticleBuffer*> * rcv_buffer_ptrs, size_t n_buffers, std::pair<ssize_t,ParticleBuffer*> p )
    {
      assert( p.first>=0 && size_t(p.first)<n_buffers );
      assert( p.second != nullptr );
      rcv_buffer_ptrs[ p.first ].exchange( p.second );
    }
    // =====================================================================

  };
  // --- end of MigrateCellParticlesNode class ---

  // helper template to avoid bug in older g++ compilers
  template<class GridT> using MigrateCellParticlesLegacyTmpl = MigrateCellParticlesLegacy<GridT>;

  // === register factory ===
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "migrate_cell_particles_legacy", make_grid_variant_operator<MigrateCellParticlesLegacyTmpl> );
  }

}

