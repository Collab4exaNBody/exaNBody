#pragma once

#include <exanb/core/particle_id_constants.h>
#include <exanb/core/basic_types.h>
#include <exanb/core/basic_types_operators.h>
#include <exanb/core/grid.h>
#include <exanb/core/grid_algorithm.h>
#include <exanb/core/particle_id_codec.h>

#include <cstdint>
#include <unordered_map>

namespace exanb
{
    using ParticleIdMap = std::unordered_multimap<uint64_t,uint64_t>;

    // find nearest particle with given id, using a ParticleIdMap. a maximum distance is given, so that a found particle further from ref than given distance will be ignored
    template<class GridT>
    static inline uint64_t global_to_nearest_local_id( uint64_t id, const ParticleIdMap& idmap, const GridT& grid, const Vec3d& ref, double max_dist)
    {
      static constexpr bool unique_instance_is_valid = false;
      
      if( id == PARTICLE_NO_ID ) { return PARTICLE_NO_ID; }

      auto range = idmap.equal_range(id);
      if( range.first == range.second ) { return PARTICLE_MISSING_ID; }
      
      auto cells = grid.cells();
      auto it = range.first;
      auto itnext = range.first;
      ++itnext;
      
      // fast-path in case only one match is found.
      // then we assume that it MUST be the particle we're looking for
      if( itnext == range.second && unique_instance_is_valid )
      {
#       ifndef NDEBUG
        size_t i=0, j=0;
        decode_cell_particle( it->second, i, j );
        assert( grid.is_valid_cell_particle(i,j) );
        const uint64_t * __restrict__ id_ptr = cells[i].field_pointer_or_null( field::id );
        if( id_ptr != nullptr ) { assert( id_ptr[j] == id ); }
        Vec3d pos = { cells[i][field::rx][j], cells[i][field::ry][j], cells[i][field::rz][j] };
        assert( norm(pos-ref) < max_dist );
#       endif
        return it->second;
      }
      
      size_t i=0, j=0;
      decode_cell_particle( it->second, i, j );
      Vec3d pos = { cells[i][field::rx][j], cells[i][field::ry][j], cells[i][field::rz][j] };
      double min_dist = norm2(pos-ref);
      uint64_t min_id = it->second;
      it = itnext;
      for (; it != range.second; ++it)
      {
        i=0, j=0;
        decode_cell_particle( it->second , i, j );
#       ifndef NDEBUG
        assert( grid.is_valid_cell_particle(i,j) );
        const uint64_t * __restrict__ id_ptr = cells[i].field_pointer_or_null( field::id );
        if( id_ptr != nullptr ) { assert( id_ptr[j] == id ); }
#       endif
        pos = Vec3d{ cells[i][field::rx][j], cells[i][field::ry][j], cells[i][field::rz][j] };
        double dist = norm2(pos-ref);
        if( dist < min_dist )
        {
          min_id = it->second;
          min_dist = dist;
        }
      }
      
      // this would very likely be a bug if found partner particle is at the same position
      // assert( min_dist > 0.0 ); // YES, but this is not this function's role to check that, it's caller's job
      
      if( min_dist < (max_dist*max_dist) ) { return min_id; }
      else { return PARTICLE_MISSING_ID; }
    }

    template<class GridT, class StreamT>
    static inline bool nearest_local_id_check( StreamT& out, uint64_t id, uint64_t local_id, const ParticleIdMap& id_map, const GridT& grid, const Vec3d& r, double max_dist)
    {
      if( ! is_particle_id_valid(local_id) )
      {
        auto cells = grid.cells();
        out << "id #"<<id<<" not found in id_map. r=" << r <<std::endl;
        auto range = id_map.equal_range(id);
        for (auto it=range.first; it!=range.second; ++it)
        {
          size_t c=0,p=0;
          decode_cell_particle( it->second , c, p );
          out << "\tcell #"<<c<<" part #"<<p;
          const uint64_t * ids = cells[c].field_pointer_or_null(field::id);
          if( grid.is_ghost_cell(c) ) { out << " (ghost)"; }
          if( ids != nullptr ) { out <<" (id="<<ids[p]<<")"; }
          Vec3d sr = { cells[c][field::rx][p] , cells[c][field::ry][p] , cells[c][field::rz][p] };
          out << " pos="<<sr << ", dist="<<norm(sr-r)<< std::endl;
        }
        out << std::flush;
        return false;
      }
      else
      {
        return true;
      }
    }

    // find local id in the central area (not in ghost layers). assumes that at most one particle can have this id in the central area.
    template<class GridT>
    static inline uint64_t global_to_own_local_id( uint64_t id, const ParticleIdMap& idmap, const GridT& grid)
    {
      if( id == PARTICLE_NO_ID ) { return PARTICLE_NO_ID; }

      auto range = idmap.equal_range(id);
      if( range.first == range.second ) { return PARTICLE_MISSING_ID; }

      auto it = range.first;
      size_t i = -1;
      size_t j = -1;
      uint64_t found_id = it->second;
      decode_cell_particle( found_id, i, j );
      bool found = ! grid.is_ghost_cell( i );
      ++it;      
      while( !found && it != range.second )
      {
        found_id = it->second;
        i = -1; j = -1;
        decode_cell_particle( found_id, i, j );
        found = ! grid.is_ghost_cell( i );
        ++it;
      }

      if( found ) { return found_id; }
      else { return PARTICLE_MISSING_ID; }
    }

}

