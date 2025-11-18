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

#include <onika/cuda/cuda_context.h>
#include <onika/yaml/yaml_utils.h>
#include <onika/log.h>

namespace exanb
{
  struct UpdateGhostConfig
  {
    onika::cuda::CudaDevice * alloc_on_device = nullptr;
    long mpi_tag = 0;
    bool gpu_buffer_pack = false;
    bool async_buffer_pack = false;
    bool staging_buffer = false;
    bool serialize_pack_send = true;
    bool wait_all = false;
    bool device_side_buffer = false;
  };
}

namespace YAML
{

  template<> struct convert< exanb::UpdateGhostConfig >
  {
    static inline bool decode(const Node& node, exanb::UpdateGhostConfig & config)
    {
      if( ! node.IsMap() )
      {
        exanb::fatal_error() << "UpdateGhostConfig must be a map" << std::endl;
        return false;
      }
      config = exanb::UpdateGhostConfig{};
      if(node["mpi_tag"])             config.mpi_tag             = node["mpi_tag"].as<int>();
      if(node["gpu_buffer_pack"])     config.gpu_buffer_pack     = node["gpu_buffer_pack"].as<bool>();
      if(node["async_buffer_pack"])   config.async_buffer_pack   = node["async_buffer_pack"].as<bool>();
      if(node["staging_buffer"])      config.staging_buffer      = node["staging_buffer"].as<bool>();
      if(node["serialize_pack_send"]) config.serialize_pack_send = node["serialize_pack_send"].as<bool>();
      if(node["wait_all"])            config.wait_all            = node["wait_all"].as<bool>();
      if(node["device_side_buffer"])  config.device_side_buffer  = node["device_side_buffer"].as<bool>();
      return true;
    }
  };

}

