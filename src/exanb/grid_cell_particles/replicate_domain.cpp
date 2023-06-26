#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/core/geometry.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/log.h>
#include <exanb/core/thread.h>

#include <vector>
#include <iostream>
#include <mpi.h>

namespace exanb
{

  template<typename GridT>
  class ReplicateDomainNode : public OperatorNode
  {
    using has_field_id_t = typename GridT:: template HasField <field::_id>;

    ADD_SLOT( MPI_Comm , mpi       , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( GridT    , grid   , INPUT_OUTPUT , REQUIRED );
    ADD_SLOT( Domain   , domain , INPUT_OUTPUT , REQUIRED );
    ADD_SLOT( IJK      , repeat , INPUT, IJK(1,1,1) );

  public:
    inline void execute () override final
    {
      MPI_Comm comm = *mpi;
      GridT& grid = *(this->grid);
      Domain& domain = *(this->domain);
      IJK repeat = *(this->repeat);

      if( repeat == IJK{1,1,1} )
      {
        grid.rebuild_particle_offsets();
        return;
      }

      int nprocs = 1;
      int rank = 0;
      MPI_Comm_rank(comm,&rank);
      MPI_Comm_size(comm,&nprocs);

      Vec3d dom_size = bounds_size(domain.bounds());
      IJK dom_dims = domain.grid_dimension();
      domain.set_bounds( { domain.bounds().bmin , domain.bounds().bmin + dom_size * repeat } );
//      domain.m_bounds.bmax = domain.m_bounds.bmin + dom_size * repeat;
      domain.set_grid_dimension( domain.grid_dimension() * repeat );

      ssize_t ghost_layers = grid.ghost_layers();
      IJK grid_dims_with_ghosts = grid.dimension();
      IJK grid_dims = grid.dimension() - (2*ghost_layers);

      GridT ngrid;
      ngrid.set_offset( grid.offset() + ghost_layers );
      ngrid.set_origin( grid.origin() );
      ngrid.set_cell_size( grid.cell_size() );
      IJK ngrid_dims = grid_dims + dom_dims * (repeat-1);
      ngrid.set_dimension( ngrid_dims );
      ngrid.set_max_neighbor_distance( 0.0 );
      ngrid.set_cell_allocator( grid.cell_allocator_ptr() );
      
      const auto & cell_allocator = ngrid.cell_allocator();
      auto src_cells = grid.cells();
      auto dst_cells = ngrid.cells();

#     ifndef NDEBUG
      size_t src_ncells = grid.number_of_cells();
      size_t dst_ncells = ngrid.number_of_cells();
#     endif

      int64_t id_max = get_id_max( has_field_id_t() );

      //size_t local_particles = 0;
#     pragma omp parallel
      {

        for(ssize_t rk=0;rk<repeat.k;rk++)
        {
          for(ssize_t rj=0;rj<repeat.j;rj++)
          {
            for(ssize_t ri=0;ri<repeat.i;ri++)
            {
              int64_t id_offset = ( rk*repeat.j*repeat.i + rj*repeat.i + ri ) * id_max;
#             pragma omp single
              ldbg << "domain duplicate @"<< IJK{ri,rj,rk}<<" id_offset="<<id_offset<<std::endl;
            
              IJK grid_displ = dom_dims * IJK{ri,rj,rk};
              Vec3d r_shift = dom_size * IJK{ri,rj,rk};
              GRID_OMP_FOR_BEGIN(grid_dims,_,cell_loc, schedule(dynamic) )
              {
                IJK src_loc = cell_loc + ghost_layers;
                assert( grid.contains( src_loc ) );

                IJK dst_loc = cell_loc + grid_displ;
                assert( ngrid.contains( dst_loc ) );

                size_t src_cell_i = grid_ijk_to_index( grid_dims_with_ghosts , src_loc );
                assert( src_cell_i>=0 && src_cell_i<src_ncells );

                size_t dst_cell_i = grid_ijk_to_index( ngrid_dims , dst_loc );
                assert( dst_cell_i>=0 && dst_cell_i<dst_ncells );

                assert( dst_cells[dst_cell_i].size() == 0 );
                size_t n = src_cells[src_cell_i].size();
                dst_cells[dst_cell_i].resize( n , cell_allocator );

                for(size_t p_i=0;p_i<n;p_i++)
                {
                  auto p = src_cells[src_cell_i][p_i];
                  p[ field::rx ] += r_shift.x;
                  p[ field::ry ] += r_shift.y;
                  p[ field::rz ] += r_shift.z;
                  apply_id_offset( p , id_offset , has_field_id_t() );
                  dst_cells[dst_cell_i].write_tuple( p_i, p );
                }
              }
              GRID_OMP_FOR_END
            }
          }
        }
        
      }

      ngrid.rebuild_particle_offsets();
      grid = std::move( ngrid );
    }

    inline int64_t get_id_max( std::true_type )
    {
      MPI_Comm comm = *mpi;
      GridT& grid = *(this->grid);
      ssize_t ghost_layers = grid.ghost_layers();
      IJK grid_dims_with_ghosts = grid.dimension();
      IJK grid_dims = grid.dimension() - (2*ghost_layers);
      auto src_cells = grid.cells();

      long id_max = 0;
#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN(grid_dims,_,cell_loc, schedule(dynamic) reduction(max:id_max) )
        {
          IJK src_loc = cell_loc + ghost_layers;
          assert( grid.contains( src_loc ) );
          size_t src_cell_i = grid_ijk_to_index( grid_dims_with_ghosts , src_loc );
          size_t n = src_cells[src_cell_i].size();
          for(size_t p_i=0;p_i<n;p_i++)
          {
            long id = src_cells[src_cell_i][field::id][p_i];
            assert( id >= 0 );
            id_max = std::max( id_max , id );
          }
        }
        GRID_OMP_FOR_END
      }
      ++ id_max;
      MPI_Allreduce( MPI_IN_PLACE , &id_max , 1 , MPI_LONG , MPI_MAX , comm);
      return id_max;
    }

    inline constexpr int64_t get_id_max( std::false_type ) { return 1; }

    template<class T>
    static inline void apply_id_offset( T& p , int64_t id_offset , std::true_type )
    {
      p[field::id] += id_offset;
    }

    template<class T>
    static inline constexpr void apply_id_offset( T& , int64_t , std::false_type ) {}


    inline void yaml_initialize(const YAML::Node& node) override final
    {
      YAML::Node tmp;
      if( node.IsSequence() && node.size()==3 )
      {
        tmp["repeat"] = node;
      }
      else { tmp = node; }
      this->OperatorNode::yaml_initialize( tmp );
    }

  };
  
   // === register factories ===
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "replicate_domain", make_grid_variant_operator< ReplicateDomainNode > );
  }

}

