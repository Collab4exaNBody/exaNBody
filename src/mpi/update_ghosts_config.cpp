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

#include <exanb/mpi/grid_update_ghosts_config.h>

namespace exanb
{
  class UpdateGhostsConfig : public OperatorNode
  {

    // -----------------------------------------------
    // Operator slots
    // -----------------------------------------------
    ADD_SLOT( GridUpdateGhostConfig , update_ghost_config , INPUT_OUTPUT , GridUpdateGhostConfig{} );

  public:
    inline void execute() override final
    {
      using onika::cuda::make_const_span;
      
      if( update_ghost_config->device_side_buffer && update_ghost_config->alloc_on_device == nullptr )
      {
        if( global_cuda_ctx()->has_devices() && global_cuda_ctx()->global_gpu_enable() )
        {
          update_ghost_config->alloc_on_device = & ( global_cuda_ctx()->m_devices[0] );
        }
      }      
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
  ONIKA_AUTORUN_INIT(update_ghosts_config)
  {
    OperatorNodeFactory::instance()->register_factory("update_ghosts_config",make_compatible_operator<UpdateGhostsConfig>);
  }

}

