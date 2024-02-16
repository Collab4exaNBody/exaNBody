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
      if(node["subcell_compaction"]) v.subcell_compaction = node["subcell_compaction"].as<bool>();
      if(node["half_symmetric"]) v.half_symmetric = node["half_symmetric"].as<bool>();
      if(node["skip_ghosts"]) v.skip_ghosts = node["skip_ghosts"].as<bool>();
      return true;
    }    
  };  
}


