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
//    bool xform_filter = true;
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
//      node["xform_filter"] = v.xform_filter;
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
//      if(node["xform_filter"]) v.xform_filter = node["xform_filter"].as<bool>();
      return true;
    }    
  };  
}


