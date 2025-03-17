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

#include <yaml-cpp/yaml.h>
#include <exanb/particle_neighbors/chunk_neighbors_specializations.h>
#include <onika/log.h>
#include <cstdlib>

namespace exanb
{
  struct ChunkNeighborsConfig
  {
    long chunk_size = 8;
    long scratch_mem_per_cell = 2048*1024; // 2 Mb scratch buffer per cell
    double stream_prealloc_factor = 1.1; // add a 10% margin to guessed best allocation size
    bool free_scratch_memory = false;
    bool build_particle_offset = false;
    bool subcell_compaction = true;
    bool half_symmetric = false;
    bool skip_ghosts = false;
    bool dual_particle_offset = false;
    bool random_access = false;
  };
}

namespace YAML
{
  template<> struct convert< exanb::ChunkNeighborsConfig >
  {
    static inline Node encode(const exanb::ChunkNeighborsConfig& v)
    {
      Node node;
      node["chunk_size"] = v.chunk_size;
      node["scratch_mem_per_cell"] = v.scratch_mem_per_cell;
      node["stream_prealloc_factor"] = v.stream_prealloc_factor;
      node["free_scratch_memory"] = v.free_scratch_memory;
      node["build_particle_offset"] = v.build_particle_offset;
      node["dual_particle_offset"] = v.dual_particle_offset;
      node["random_access"] = v.random_access;
      node["subcell_compaction"] = v.subcell_compaction;
      node["half_symmetric"] = v.half_symmetric;
      node["skip_ghosts"] = v.skip_ghosts;
      return node;
    }
    static inline bool decode(const Node& node, exanb::ChunkNeighborsConfig& v)
    {
      if( ! node.IsMap() ) { return false; }
      v = exanb::ChunkNeighborsConfig{};
      if(node["chunk_size"]) v.chunk_size = node["chunk_size"].as<long>();
      if(node["scratch_mem_per_cell"]) v.scratch_mem_per_cell = node["scratch_mem_per_cell"].as<long>();
      if(node["stream_prealloc_factor"]) v.stream_prealloc_factor = node["stream_prealloc_factor"].as<double>();
      if(node["free_scratch_memory"]) v.free_scratch_memory = node["free_scratch_memory"].as<bool>();
      if(node["build_particle_offset"]) v.build_particle_offset = node["build_particle_offset"].as<bool>();
      if(node["dual_particle_offset"]) v.dual_particle_offset = node["dual_particle_offset"].as<bool>();
      if(node["random_access"]) v.random_access = node["random_access"].as<bool>();
      if(node["subcell_compaction"]) v.subcell_compaction = node["subcell_compaction"].as<bool>();
      if(node["half_symmetric"]) v.half_symmetric = node["half_symmetric"].as<bool>();
      if(node["skip_ghosts"]) v.skip_ghosts = node["skip_ghosts"].as<bool>();
      
      if( v.dual_particle_offset && !v.build_particle_offset )
      {
        exanb::lerr << "Warning: dual_particle_offset requires build_particle_offset. build_particle_offset enabled"<<std::endl;
        v.build_particle_offset = true;
      }

      if( v.random_access && v.chunk_size != 1 )
      {
        exanb::lerr << "Warning: random_access requires chunk_size to be forced to 1"<<std::endl;
        v.chunk_size = 1;
      }
      
      [[maybe_unused]] const unsigned int VARIMPL = v.chunk_size;
      int nearest = -1;
#     define _XNB_CHUNK_NEIGHBORS_CS_NEAREST( CS ) if( std::abs( static_cast<int>(v.chunk_size) - static_cast<int>(CS) ) < std::abs(int(v.chunk_size)-int(nearest)) ) nearest = CS;
      XNB_CHUNK_NEIGHBORS_CS_SPECIALIZE( _XNB_CHUNK_NEIGHBORS_CS_NEAREST )
#     undef _XNB_CHUNK_NEIGHBORS_CS_NEAREST
      
      if( nearest != v.chunk_size )
      {
        exanb::lerr << "Warning: chunk_size="<<v.chunk_size<<" is not supported by this version, setting to nearest available ("<<nearest<<")"<<std::endl;
        v.chunk_size = nearest;
      }

      if( v.random_access && v.chunk_size != 1 )
      {
        exanb::lerr << "random_access and/or dual_particle_offset is only possible with chunk_size=1"<<std::endl;
        return false;
      }

/*
      std::cout << "chunk_size             = " << v.chunk_size <<std::endl;
      std::cout << "scratch_mem_per_cell   = " << v.scratch_mem_per_cell <<std::endl;
      std::cout << "stream_prealloc_factor = " << v.stream_prealloc_factor <<std::endl;
      std::cout << "free_scratch_memory    = " << v.free_scratch_memory <<std::endl;
      std::cout << "build_particle_offset  = " << v.build_particle_offset <<std::endl;
      std::cout << "dual_particle_offset   = " << v.dual_particle_offset <<std::endl;
      std::cout << "random_access          = " << v.random_access <<std::endl;
      std::cout << "subcell_compaction     = " << v.subcell_compaction <<std::endl;
      std::cout << "half_symmetric         = " << v.half_symmetric <<std::endl;
      std::cout << "skip_ghosts            = " << v.skip_ghosts <<std::endl;
*/
      return true;
    }    
  };  
}


