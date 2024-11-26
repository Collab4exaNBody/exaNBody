/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/

#pragma once

#include <onika/thread.h>
#include <onika/memory/allocator.h>
#include <onika/cuda/cuda.h>
#include <onika/cuda/stl_adaptors.h>
#include <onika/math/basic_types.h>

#include <exanb/fields.h>
#include <exanb/field_sets.h>
#include <exanb/core/grid_algorithm.h>
#include <exanb/core/cell_particles_from_field_set.h>
#include <exanb/core/grid_cell_compute_profiler.h>
#include <exanb/core/grid_particle_field_accessor.h>
#include <exanb/core/flat_arrays.h>
#include <exanb/core/grid_particle_field_accessor.h>

#include <cstdlib>
#include <vector>
#include <type_traits>
#include <memory>

#include <iostream>

#include <onika/soatl/field_arrays.h>
#include <onika/memory/memory_usage.h>

#ifndef XSTAMP_FIELD_ARRAYS_STORE_COUNT
#define XSTAMP_FIELD_ARRAYS_STORE_COUNT 3
#endif

namespace exanb
{

  // simple alias for array of lock arrays
  using onika::spin_mutex_array;
  using GridParticleLocks = std::vector<spin_mutex_array>;

  template<class GridFieldSet> class Grid;

  template<typename... particle_field_ids>
  class Grid< FieldSet<particle_field_ids...> > 
  {
    static constexpr double c_epsilon = 1.0 / (1ull<<48) ; // approx. 3.5e-15

  public:

    static constexpr size_t MaxStoredPointerCount = XSTAMP_FIELD_ARRAYS_STORE_COUNT;
    static constexpr size_t StoredPointerCount = std::min( MaxStoredPointerCount , size_t(sizeof...(particle_field_ids)+3) );

    using field_set_t = FieldSet<particle_field_ids...>;
    static inline constexpr field_set_t field_set{};
    
    using CellParticlesAllocator = onika::soatl::PackedFieldArraysAllocatorImpl< onika::memory::DefaultAllocator, onika::memory::DEFAULT_ALIGNMENT, onika::memory::DEFAULT_CHUNK_SIZE, field::_rx,field::_ry,field::_rz, particle_field_ids... > ;
    using CellParticles = onika::soatl::FieldArraysWithAllocator< onika::memory::DEFAULT_ALIGNMENT, onika::memory::DEFAULT_CHUNK_SIZE, CellParticlesAllocator, StoredPointerCount , field::_rx,field::_ry,field::_rz, particle_field_ids... >;
    static_assert( std::is_same_v< CellParticles , cell_particles_from_field_set_t<field_set_t> > , "CellParticles type not as expected" );

    static constexpr size_t Alignment = CellParticles::Alignment;
    static constexpr size_t ChunkSize = CellParticles::ChunkSize;
    using Fields = FieldSet< field::_rx,field::_ry,field::_rz,particle_field_ids...>;
    using ParticleTuple = onika::soatl::FieldTuple<field::_rx,field::_ry,field::_rz,particle_field_ids...>;
    template<class _id> using HasField = typename CellParticles::template HasField < _id > ;

    // Grid's origin shall always be equal to domain's lower boundary.
    inline void set_origin(Vec3d o) { m_origin = o; }
    inline Vec3d origin() const { return m_origin; }

    inline void set_cell_size(double s) { m_cell_size = s; }
    ONIKA_HOST_DEVICE_FUNC inline double cell_size() const { return m_cell_size; }

    // quantities usefull for testing
//    inline double cell_size2() const { return m_cell_size*m_cell_size; }
    ONIKA_HOST_DEVICE_FUNC inline double epsilon_cell_size() const { return c_epsilon * cell_size(); }
    ONIKA_HOST_DEVICE_FUNC inline double epsilon_cell_size2() const { double x=epsilon_cell_size(); return x*x; }

    ONIKA_HOST_DEVICE_FUNC inline IJK dimension() const
    {
      return m_dimension;
    }

    // offset tells where in the domain's global grid this grid is located (i.e. the local grid's lower cell location in the domain grid)
    inline void set_offset(IJK offset) { m_offset = offset; }
    inline IJK offset() const { return m_offset; }

    inline GridBlock block() const { return GridBlock{ m_offset , IJK{m_offset.i+m_dimension.i,m_offset.j+m_dimension.j,m_offset.k+m_dimension.k} }; }

    inline void set_max_neighbor_distance(double rmax) { m_max_neighbor_distance = rmax; }
    inline double max_neighbor_distance() const { return m_max_neighbor_distance; }    
    inline size_t ghost_layers() const { return static_cast<size_t>( std::ceil( max_neighbor_distance() / cell_size() ) ); }

    // get start position of a cell
    ONIKA_HOST_DEVICE_FUNC inline Vec3d cell_position(const IJK& loc) const 
    {
      return m_origin+((m_offset+loc)*m_cell_size);
    }

    // get spatial bounds of a cell
    ONIKA_HOST_DEVICE_FUNC inline AABB cell_bounds(const IJK& loc) const
    {
      return AABB{ this->cell_position(loc) , this->cell_position(loc+1) };
    }

    // locate a cell from a point in space. cell location returned contains the point p.
    // return the cell coordinate relative to the local grid. e.g. if a particle is contained is the cell at the low corner of this grid, the retruned loc is IJK{0,0,0}
    ONIKA_HOST_DEVICE_FUNC inline IJK locate_cell(const Vec3d& p_in) const
    {
      Vec3d p = ( p_in - m_origin ) / m_cell_size;
      IJK loc = make_ijk( p ); // this is correct because make_ijk(Vec3d) apply std::floor on x,y,z components. a simple cast to an int does not work because of nearest rounding.
      loc = loc - m_offset;
      assert( is_inside_threshold( cell_bounds(loc) , p_in , epsilon_cell_size2() ) ); // approximative
      //assert( is_inside( cell_bounds(loc) , p_in ) ); // strict
      return loc;
    }

    // loc is relative (to grid) IJK coordinate.
    inline bool contains(const IJK& loc) const
    {
      return grid_contains(m_dimension,loc);
    }

    inline const size_t * cell_particle_offset_data() const
    {
      assert( m_cell_particle_offset.size() == number_of_cells()+1 ); 
      return m_cell_particle_offset.data();
    }

    // valid only if rebuild_particle_offsets has been called before (and after the last change of number of particles in at least one cell)
    inline size_t cell_particle_offset(size_t cell_i) const
    {
      assert(cell_i<m_cell_particle_offset.size()); 
      return m_cell_particle_offset[cell_i]; 
    }

    // return the global index of particle (e.g. it's unique id across the grid, not its cell's local index)
    inline size_t particle_index(size_t cell_i, size_t p_j) const
    {
      assert(cell_i<m_cell_particle_offset.size());
      return m_cell_particle_offset[cell_i]+p_j;
    }

    // valid only if rebuild_particle_offsets has been called before (and after the last change of number of particles in at least one cell)
    inline size_t number_of_particles() const
    {
      assert(m_cell_particle_offset.size()==(number_of_cells()+1)); 
      return m_cell_particle_offset[number_of_cells()]; 
    }

    // get spatial bounds of the grid
    inline AABB grid_bounds() const 
    { 
      return AABB{ m_origin+(m_offset*m_cell_size) , m_origin+((m_offset+m_dimension)*m_cell_size) }; 
    }

    // get particle parametric coords inside cell
    inline Vec3d particle_pcoord(const IJK& cell, const Vec3d& r) const
    {
      Vec3d cell_low = m_origin+((m_offset+cell)*m_cell_size);
      Vec3d pcoord = ( r - cell_low ) / m_cell_size;
      // assert( is_inside( AABB{ {0.,0.,0.} , {1.,1.,1.} } , pcoord ) );
      return pcoord;
    }

    inline IJK cell_ijk(size_t index) const
    {
      return grid_index_to_ijk(m_dimension, index);
    }

    inline size_t cell_index(const IJK& loc) const
    {
      return grid_ijk_to_index( m_dimension, loc );
    }
    
    inline bool is_ghost_cell(const IJK& cell_pos) const
    {
      size_t glayers = ghost_layers();
      return inside_grid_shell(m_dimension, 0, glayers, cell_pos );
    }

    inline bool is_ghost_cell(size_t index) const
    {
      return is_ghost_cell( cell_ijk(index) );
    }

    inline size_t number_of_ghost_cells() const
    {
      IJK dims = dimension();
      ssize_t gl = ghost_layers();
      IJK dimsNoGhost = { dims.i-2*gl , dims.j-2*gl , dims.k-2*gl };
      return ( dims.i*dims.j*dims.k ) - ( dimsNoGhost.i*dimsNoGhost.j*dimsNoGhost.k );
    }

    //! number of ghosts particles i.e. number of particles in ghost cells
    inline size_t number_of_ghost_particles() const
    {
      size_t nb_particles_ghosts = 0;
      size_t n_cells = number_of_cells();
      for(size_t i=0; i<n_cells; i++)
        {
          if(is_ghost_cell(i))
            {
              nb_particles_ghosts += cell_number_of_particles(i);
            }
        }
      return nb_particles_ghosts;
    }


    // set grid dimensions
    inline void set_dimension(IJK dims)
    {
      m_dimension = dims;
      m_cells.resize( grid_cell_count(m_dimension) );
      update_cell_profiling_storage(); 
    }

    // return number if cells in grid (including ghost layers)
    ONIKA_HOST_DEVICE_FUNC inline size_t number_of_cells() const { return onika::cuda::vector_size( m_cells ); }

    // cell's particles data allocator
    inline const onika::soatl::PackedFieldArraysAllocator & cell_allocator() const
    {
      static const CellParticlesAllocator default_allocator( onika::memory::DefaultAllocator{ onika::memory::CUDA_FALLBACK_ALLOC_POLICY } );
      if( m_cell_allocator != nullptr ) return *m_cell_allocator;
      else return default_allocator;
    }
    inline std::shared_ptr<onika::soatl::PackedFieldArraysAllocator> cell_allocator_ptr()
    {
      return m_cell_allocator;
    }
    inline void set_cell_allocator( std::shared_ptr<onika::soatl::PackedFieldArraysAllocator> a )
    {
      m_cell_allocator = a;
    }
    
    template<class... ids> inline void set_cell_allocator_for_fields( FieldSet<ids...> )
    {
      using onika::soatl::pfa_allocator_impl_from_field_ids_t;
      using onika::soatl::FieldIdsIncludingSequence;
      using onika::soatl::FieldIds;
      using onika::memory::GenericHostAllocator;
      using RefFieldIds = FieldIds<field::_rx,field::_ry,field::_rz,particle_field_ids...>;
      using ReqFieldIds = FieldIds<ids...>;
      using FilteredSubSet = typename FieldIdsIncludingSequence< RefFieldIds , ReqFieldIds >::type ;
      using AllocT = pfa_allocator_impl_from_field_ids_t< onika::memory::DefaultAllocator, onika::memory::DEFAULT_ALIGNMENT, onika::memory::DEFAULT_CHUNK_SIZE, FilteredSubSet >;
      m_cell_allocator = std::make_shared< AllocT >();
      m_cell_allocator->set_gpu_addressable_allocation( GenericHostAllocator::cuda_enabled() );
    }
    
    ONIKA_HOST_DEVICE_FUNC inline       CellParticles* cells()       { return onika::cuda::vector_data(m_cells); }
    ONIKA_HOST_DEVICE_FUNC inline const CellParticles* cells() const { return onika::cuda::vector_data(m_cells); }

    ONIKA_HOST_DEVICE_FUNC inline auto cells_accessor()       { return make_cells_accessor      ( onika::cuda::vector_data(m_cells) , onika::cuda::vector_data(m_cell_particle_offset) ); }
    ONIKA_HOST_DEVICE_FUNC inline auto cells_accessor() const { return make_cells_const_accessor( onika::cuda::vector_data(m_cells) , onika::cuda::vector_data(m_cell_particle_offset) ); }
    
    inline       CellParticles& cell(IJK loc)       { return m_cells[grid_ijk_to_index(m_dimension,loc)]; }
    inline const CellParticles& cell(IJK loc) const { return m_cells[grid_ijk_to_index(m_dimension,loc)]; }

    inline       CellParticles& cell(size_t index)       { return m_cells[index]; }
    inline const CellParticles& cell(size_t index) const { return m_cells[index]; }

    // access cell and particle geometry information
    inline Vec3d particle_position(size_t c, size_t p) const
    {
      auto t = m_cells[c][p]; return Vec3d{ t[field::rx], t[field::ry], t[field::rz] };
    }

    [[deprecated]] inline int particle_type(size_t c, size_t p) const
    {
#     ifdef EXANB_PARTICLE_TYPE_FIELD
      const auto * __restrict__ type_ptr = m_cells[c].field_pointer_or_null( EXANB_PARTICLE_TYPE_FIELD );
      return type_ptr[p];
#     else
      return 0;
#     endif
    }

    inline size_t cell_number_of_particles(size_t cell_i) const
    {
      assert( /* cell_i>=0 && */ cell_i<number_of_cells() );
      return m_cells[cell_i].size();
    }

    inline bool cell_is_gpu_addressable(size_t cell_i) const
    {
      return m_cells[cell_i].is_gpu_addressable();
    }

    // compute global particle offset (scan sum of particle count in cells)
    // array must be allocated with at least this->number_of_cells() elements
    // first element is the number of particle in cell #0, 2nd element is number of particles in cell #0 + number of paricles in cell #1, etc.
    // last element, in array[this->number_of_cells()-1], equals the total number of particles
    inline void rebuild_particle_offsets()
    {
      const size_t n_cells = number_of_cells();
      m_cell_particle_offset.resize( n_cells + 1 );
      size_t total_particles = 0;
      for(size_t i=0;i<n_cells;i++)
      {        
        m_cell_particle_offset[i] = total_particles;
        total_particles += cell_number_of_particles(i);
      }
      m_cell_particle_offset.back() = total_particles;

      // also rebuild particle ghost flag array
      m_ghost_particle.assign( (total_particles+63)/64 , 0 );
      for(size_t i=0;i<n_cells;i++)
      {
        if( is_ghost_cell(i) )
        {
          size_t n_particles = m_cells[i].size();
          for(size_t j=0;j<n_particles;j++)
          {
            size_t pidx = m_cell_particle_offset[i] + j;
            m_ghost_particle[ pidx / 64 ] |= 1ull << ( pidx % 64 );
          }
        }
      }
      m_ghost_particle_ptr = m_ghost_particle.data();

#     ifndef NDEBUG
      assert( m_cell_particle_offset[0] == 0 );
      assert( m_cell_particle_offset[n_cells] == total_particles );
      for(size_t i=0;i<n_cells;i++)
      {
        assert( cell_number_of_particles(i) == ( m_cell_particle_offset[i+1] - m_cell_particle_offset[i] ) );
      }
#     endif
    }
    
    ONIKA_HOST_DEVICE_FUNC
    inline bool is_ghost_particle( size_t pidx )
    {
      return ( m_ghost_particle_ptr[ pidx / 64 ] & ( 1ull << ( pidx % 64 ) ) ) != 0;
    }
    
    ONIKA_HOST_DEVICE_FUNC
    inline const uint64_t * __restrict__ particle_ghost_flag_data() const
    {
      return m_ghost_particle_ptr;
    } 

    inline void clear_particles()
    {
      for(auto& cell : m_cells) { cell.clear( cell_allocator() ); }
      m_cells.clear();
      m_cells.shrink_to_fit();
      rebuild_particle_offsets();
    }

    inline void reset()
    {
      m_offset = IJK{0,0,0};
      m_dimension = IJK{0,0,0};
      m_origin = Vec3d{0.,0.,0.};
      m_cell_size = 1.;
      m_max_neighbor_distance = 0.;
      clear_particles();
    }

    inline bool is_valid_cell_particle(size_t cell_index, size_t part_index) const
    {
      if( ssize_t(cell_index)>=0 && cell_index<m_cells.size() && ssize_t(part_index)>=0 )
      {
        return part_index < m_cells[cell_index].size();
      }
      else
      {
        return false;
      }
    }

    // memory consumption estimate
    inline size_t memory_bytes() const
    {
      size_t m = sizeof(*this)
               + m_cell_particle_offset.capacity() * sizeof(size_t)
               + m_cells.capacity() * sizeof(CellParticles)
               + m_grid_compute_profiling.capacity() * sizeof(GridCellComputeProfiling);
      size_t n_cells = m_cells.size();
      for(size_t i=0;i<n_cells;i++)
      {
        m += m_cells[i].storage_size( cell_allocator() );
      }
      return m;
    }

    inline void check_cells_are_gpu_addressable() const
    {
#     ifndef NDEBUG
      assert( cell_allocator().allocates_gpu_addressable() );
      const size_t N = number_of_cells();
      size_t N_non_gpu = 0;
      for(size_t i=0;i<N;i++)
      {
        if( ! m_cells[i].is_gpu_addressable( cell_allocator() ) )
        {
          auto loc = cell_ijk(i);
          std::cerr<<"Cell #"<<i<<" @("<<loc.i<<","<<loc.j<<","<<loc.k<<") " << (is_ghost_cell(i)?" (ghost)":"") << ", with "<<m_cells[i].size()<<" particles, is not GPU adressable\n" << std::flush;
	        ++ N_non_gpu;
        }
      }
      if( N_non_gpu )
      {
        std::cerr<<N_non_gpu<<"/"<<N<<" cells are not GPU adressable\n" << std::flush;
        std::abort();
      }
#     endif
    }

    // *************** copy/move operator/constructor **************

    Grid() = default;
    Grid( const Grid & other ) = default;
    inline Grid( Grid && other )
      : m_offset( other.m_offset )
      , m_dimension( other.m_dimension )
      , m_origin( other.m_origin )
      , m_cell_size( other.m_cell_size )
      , m_max_neighbor_distance( other.m_max_neighbor_distance )
      , m_cell_particle_offset( std::move( other.m_cell_particle_offset ) )
      , m_cells( std::move( other.m_cells ) )
      , m_cell_allocator( std::move( other.m_cell_allocator ) )
    {
      other.m_dimension = IJK{0,0,0};
      other.m_max_neighbor_distance = 0;
      assert( other.m_cell_particle_offset.size() == 0 );
      other.m_cell_allocator = nullptr; // just in case
      assert( other.m_cells.size() == 0 );
    }

    inline ~Grid()
    {
      for(auto& cell : m_cells) { cell.clear( cell_allocator() ); }      
    }

    inline Grid& operator = ( const Grid & other )
    {
      for(auto& cell : m_cells) { cell.clear( cell_allocator() ); }
      m_offset = other.m_offset;
      m_dimension = other.m_dimension;
      m_origin = other.m_origin;
      m_cell_size = other.m_cell_size;
      m_max_neighbor_distance = other.m_max_neighbor_distance;
      m_cell_particle_offset = other.m_cell_particle_offset;
      m_cell_allocator = other.m_cell_allocator;
      size_t n_cells = other.m_cells.size();
      m_cells.resize( n_cells );
#     pragma omp parallel for schedule(dynamic)
      for(size_t i=0;i<n_cells;i++)
      {
        m_cells[i].copy_from( other.m_cells[i] , cell_allocator() );
      }
      return *this;
    }
    inline Grid& operator = ( Grid && other )
    {
      for(auto& cell : m_cells) { cell.clear( cell_allocator() ); }
      m_offset = other.m_offset;
      m_dimension = other.m_dimension;
      m_origin = other.m_origin;
      m_cell_size = other.m_cell_size;
      m_max_neighbor_distance = other.m_max_neighbor_distance;
      m_cell_particle_offset = std::move( other.m_cell_particle_offset );
      m_cell_allocator = std::move( other.m_cell_allocator );
      m_cells = std::move( other.m_cells );
      other.m_dimension = IJK{0,0,0};
      other.m_max_neighbor_distance = 0;
      other.m_cell_allocator = nullptr;
      assert( other.m_cells.size() == 0 );
      assert( other.m_cell_particle_offset.size() == 0 );
      return *this;
    }

    // ************** per cell compute profiling **************
    inline void set_cell_profiling(bool onOff) 
    {
#     ifdef XNB_GRID_CELL_COMPUTE_PROFILING
      m_enable_per_cell_profiling = onOff;
      update_cell_profiling_storage();
#     else
      m_enable_per_cell_profiling = false && onOff;
#     endif
    }
    
    inline bool cell_profiling() const
    {
      return m_enable_per_cell_profiling;
    }

    inline void update_cell_profiling_storage()
    {
      if( m_enable_per_cell_profiling )
      {
        m_grid_compute_profiling.resize( number_of_cells() );
      }
      else
      {
        m_grid_compute_profiling.clear();
        m_grid_compute_profiling.shrink_to_fit();
      }
    }

    inline void reset_cell_profiling_data()
    {
      for( auto& x : m_grid_compute_profiling) { x.m_time = 0.0; }
    }
    
    inline GridCellComputeProfiler cell_profiler()
    {
      if( m_enable_per_cell_profiling ) return { m_grid_compute_profiling.data() };
      else return { nullptr };
    }
    // ********************************************************


    // ***************** optional flat arrays *****************
    template<class fid>
    inline typename onika::soatl::FieldId<fid>::value_type * flat_array_data( onika::soatl::FieldId<fid> )
    {
      auto it = m_flat_arrays.find( onika::soatl::FieldId<fid>::short_name() );
      if( it == m_flat_arrays.end() )
      {        
        it = m_flat_arrays.insert( { onika::soatl::FieldId<fid>::short_name() , std::make_shared< FlatArrayAdapter<fid> >() } ).first;
      }
      assert( it != m_flat_arrays.end() );
      FlatArrayAdapter<fid> * fa = reinterpret_cast< FlatArrayAdapter<fid> * >( it->second.get() );
      assert( fa != nullptr );
      assert( fa->type_name() == typeid( typename onika::soatl::FieldId<fid>::value_type ).name() );
      fa->resize( number_of_particles() );
      return fa->data();
    }
    template<class fid>
    inline auto flat_array_accessor( onika::soatl::FieldId<fid> f )
    {
      using details::OptionalCellParticleFieldAccessor;
      return OptionalCellParticleFieldAccessor< onika::soatl::FieldId<fid> , false > { flat_array_data(f) };
    }
    template<class fid>
    inline auto flat_array_const_accessor( onika::soatl::FieldId<fid> f )
    {
      using details::OptionalCellParticleFieldAccessor;
      return OptionalCellParticleFieldAccessor< onika::soatl::FieldId<fid> , true >{ flat_array_data(f) };
    }
    inline auto remove_flat_array( const std::string& name )
    {
      m_flat_arrays.erase( name );
    }
    inline void clear_flat_array( const std::string& name )
    {
      auto it = m_flat_arrays.find( name );
      if( it != m_flat_arrays.end() ) it->second->clear();
    }
    inline void shrink_flat_array( const std::string& name )
    {
      auto it = m_flat_arrays.find( name );
      if( it != m_flat_arrays.end() ) it->second->shrink_to_fit();
    }
    // ********************************************************


    // *** unification of per cell arrays dans flat arrays ***
    template<class fid>
    inline auto
    field_accessor( onika::soatl::FieldId<fid> f )
    {
      if constexpr ( ! HasField<fid>::value ) return flat_array_accessor(f);
      if constexpr (   HasField<fid>::value ) return f;
      // we shall never get there, but intel compiler needs this to avoid compile warnings
      return std::conditional_t< HasField<fid>::value , onika::soatl::FieldId<fid> , decltype(flat_array_accessor(f)) >{} ;
    }
    template<class fid>
    inline auto field_const_accessor( onika::soatl::FieldId<fid> f )
    {
      if constexpr ( ! HasField<fid>::value ) return flat_array_const_accessor(f);
      if constexpr (   HasField<fid>::value ) return f;
      // we shall never get there, but intel compiler needs this to avoid compile warnings
      return std::conditional_t< HasField<fid>::value , onika::soatl::FieldId<fid> , decltype(flat_array_const_accessor(f)) >{} ; // should never get ther
    }
    template<class... fids>
    inline auto field_accessors_from_field_set( FieldSet<fids...> fs )
    {
      return onika::make_flat_tuple( field_accessor( onika::soatl::FieldId<fids>{} ) ... );
    }
    template<class fid>
    inline bool has_allocated_field( onika::soatl::FieldId<fid> f )
    {
      return HasField<fid>::value || m_flat_arrays.find( onika::soatl::FieldId<fid>::short_name() ) != m_flat_arrays.end() ;
    }    
    template<class... fids>
    inline auto has_allocated_fields( FieldSet<fids...> )
    {
      return ( ... && ( has_allocated_field( onika::soatl::FieldId<fids>{} ) ) );
    }
    // ******************************************************* 

  private:
    IJK m_offset = {0,0,0};
    IJK m_dimension = {0,0,0};
    Vec3d m_origin = {0.,0.,0.};
    double m_cell_size = 1.;
    double m_max_neighbor_distance = 0.;
    onika::memory::CudaMMVector<size_t> m_cell_particle_offset;
    onika::memory::CudaMMVector< uint64_t > m_ghost_particle; // size = number of particles / 64 (rounded to upper 64 multiple)
    uint64_t const * __restrict__ m_ghost_particle_ptr = nullptr;
      
    // storage of particles split into cubic cells.
    onika::memory::CudaMMVector< CellParticles > m_cells;
    std::shared_ptr<onika::soatl::PackedFieldArraysAllocator> m_cell_allocator;

    // optional arrays, stored as "flat" arrays, dynamically allocated
    // these are scratch storage, they're content are lost every time rebuild_particle_offsets is called
    std::unordered_map< std::string , std::shared_ptr<FlatArrayDescriptor> > m_flat_arrays;
    
    // per grid profiling information, optional.
    GridComputeProfiling m_grid_compute_profiling;
    bool m_enable_per_cell_profiling = false;
  }; // end of class Grid



  // =============================================================
  // =================== utility functions =======================
  // =============================================================

  // factory to instatiate a grid
  template<typename... ids>
  static inline Grid< FieldSet<ids...> > make_grid( const onika::soatl::FieldId<ids>& ... )
  {
    return Grid< FieldSet<ids...> >();
  }

  // utility template to instantiate a grid with particle fields specified in a exanb::FieldSet
  template<typename field_set> struct GridFromFieldSetHelper {};
  template<typename... field_ids > struct GridFromFieldSetHelper< FieldSet<field_ids...> > { using type = Grid< FieldSet<field_ids...> >; };

  template<typename field_set>
  using GridFromFieldSet = typename GridFromFieldSetHelper<field_set>::type ;

  template<typename GridT, class field_id>
  using GridHasField = typename GridT::CellParticles:: template HasField<field_id> ;
  
  template<class GridT, class field_id> static inline constexpr bool grid_has_field_v = GridHasField<GridT,field_id>::value;

  template<typename GridT, typename field_id>
  using AssertGridHasField = std::enable_if_t< GridHasField<GridT,field_id>::value >;

  template<bool... X> struct AllTrueHelper : public std::true_type {};
  template<bool X, bool... Y> struct AllTrueHelper<X,Y...> : public std::integral_constant<bool, X && AllTrueHelper<Y...>::value > {};

  template<typename GridT, typename... field_ids>
  using GridHasFields = std::integral_constant<bool, AllTrueHelper< GridHasField<GridT,field_ids>::value ... >::value > ;

  template<typename GridT, typename... field_ids>
  using AssertGridHasFields = std::enable_if_t< GridHasFields<GridT,field_ids...>::value >;

  template<typename GridT, typename FieldSetT > struct GridContainFieldSet;
  template<typename GridT, typename... field_ids > struct GridContainFieldSet< GridT, FieldSet<field_ids...> > : public GridHasFields<GridT,field_ids...> {};
  template<class GridT, class FieldSetT > static inline constexpr bool grid_contains_field_set_v = GridContainFieldSet<GridT,FieldSetT>::value;

  template<typename GridT, typename FieldSetT>
  using AssertGridContainFieldSet = std::enable_if_t< GridContainFieldSet<GridT,FieldSetT>::value >;


  template<typename GridT, typename FS> struct _GridFieldSetPointerTuple {};
  template<typename GridT, typename... field_ids>
  struct _GridFieldSetPointerTuple< GridT , FieldSet<field_ids...> >
  {
    using PointerTuple = onika::soatl::FieldPointerTuple<GridT::CellParticles::Alignment,GridT::CellParticles::ChunkSize,field_ids...>;
  };
  template<typename GridT, typename FS>
  using GridFieldSetPointerTuple = typename _GridFieldSetPointerTuple<GridT,FS>::PointerTuple;

  template<class GridT, class FuncT, class... FieldsOrCombiners>
  static inline void apply_grid_fields(GridT& grid, FuncT f, const FieldsOrCombiners& ... fc )
  {
    ( ... , ( f( grid, fc ) ) ) ;
  }

  template<class GridT, class FuncT, class... fid>
  static inline void apply_grid_field_set(GridT& grid, FuncT f, FieldSet<fid...> )
  {
    apply_grid_fields( grid , f , onika::soatl::FieldId<fid>{} ... );
  }

  template<class GridT, class FuncT, class FieldTupleT, size_t... FieldIndex >
  static inline void apply_grid_field_tuple_idx(GridT& grid, FuncT f, const FieldTupleT & tp , std::index_sequence<FieldIndex...> )
  {
    apply_grid_fields( grid , f , tp.get( onika::tuple_index_t<FieldIndex>{} ) ... );
  }

  template<class GridT, class FuncT, class FieldTupleT >
  static inline void apply_grid_field_tuple(GridT& grid, FuncT f, const FieldTupleT & tp)
  {
    apply_grid_field_tuple_idx( grid, f, tp, std::make_index_sequence< onika::tuple_size_const_v<FieldTupleT> >{} );
  }
  
  template<class... FS> inline std::vector<std::string> xnb_grid_variants_as_strings( FieldSets<FS...> ) { return { onika::pretty_short_type<FS>() ... }; }
  inline std::vector<std::string> xnb_grid_variants_as_strings() { return xnb_grid_variants_as_strings( standard_field_sets_v ); }

} // end of namespace exanb


