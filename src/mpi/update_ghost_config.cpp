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

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>

#include <exanb/mpi/update_ghost_config.h>

namespace exanb
{
  class UpdateGhostConfigOperator : public OperatorNode
  {

    // -----------------------------------------------
    // Operator slots
    // -----------------------------------------------
    ADD_SLOT( UpdateGhostConfig , update_ghost_config , INPUT_OUTPUT , UpdateGhostConfig{} );
    ADD_SLOT( bool , verbose , INPUT , true );
    
  public:
    inline void execute() override final
    {
      if( global_cuda_ctx() == nullptr || ! global_cuda_ctx()->has_devices() || ! global_cuda_ctx()->global_gpu_enable() )
      {
        update_ghost_config->gpu_buffer_pack = false;
        update_ghost_config->alloc_on_device = nullptr;
      }
      if( ! update_ghost_config->gpu_buffer_pack )
      {
        update_ghost_config->staging_buffer = false;
      }
      if( update_ghost_config->gpu_buffer_pack && update_ghost_config->alloc_on_device == nullptr )
      {
        update_ghost_config->alloc_on_device = & ( global_cuda_ctx()->m_devices[0] );
      }
      
      if( *verbose )
      {
        lout << "============ Update Ghost Configuration =========" << std::endl
             << "mpi_tag             = "<<update_ghost_config->mpi_tag<< std::endl
             << "gpu_buffer_pack     = "<<std::boolalpha << update_ghost_config->gpu_buffer_pack<< std::endl
             << "async_buffer_pack   = "<<std::boolalpha << update_ghost_config->async_buffer_pack<< std::endl
             << "staging_buffer      = "<<std::boolalpha << update_ghost_config->staging_buffer<< std::endl
             << "serialize_pack_send = "<<std::boolalpha << update_ghost_config->serialize_pack_send<< std::endl
             << "wait_all            = "<<std::boolalpha << update_ghost_config->wait_all<< std::endl
             << "alloc_on_device     = "<< (void*) update_ghost_config->alloc_on_device << std::endl
             << "================================================="<< std::endl << std::endl;
      }
    }

    inline void yaml_initialize(const YAML::Node& node) override final
    {
      YAML::Node tmp;
      if( ! node["update_ghost_config"] )
      {
        tmp["update_ghost_config"] = node;
      }
      else { tmp = node; }
      this->OperatorNode::yaml_initialize(tmp);
    }

    inline std::string documentation() const override final
    {
      return R"EOF(

Initializes and outputs a configuration strucure for subsequent update_ghosts and update_from_ghosts nodes.

Usage example:

update_ghosts_config:
  mpi_tag: 0
  gpu_buffer_pack: true
  async_buffer_pack: true
  staging_buffer: false
  serialize_pack_send: true
  wait_all: false
  device_side_buffer: true
  
)EOF";
    }    
  };

  // === register factory ===
  ONIKA_AUTORUN_INIT(update_ghost_config)
  {
    OperatorNodeFactory::instance()->register_factory("update_ghost_config",make_compatible_operator<UpdateGhostConfigOperator>);
  }

}

