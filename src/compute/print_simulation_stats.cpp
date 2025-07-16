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
#include <onika/log.h>
#include <onika/cpp_utils.h>
#include <exanb/core/domain.h>

#include <exanb/compute/simulation_statistics.h>

namespace exanb
{
  using onika::scg::OperatorNodeFactory;
  using onika::scg::OperatorNode;

  class PrintSimulationStatistics : public OperatorNode
  {  
    // thermodynamic state & physics data
    ADD_SLOT( long               , timestep            , INPUT , REQUIRED );
    ADD_SLOT( double             , physical_time       , INPUT , REQUIRED );
    ADD_SLOT( SimulationStatistics    , simulation_stats    , INPUT , REQUIRED );

  public:
    inline bool is_sink() const override final { return true; }
  
    inline void execute () override final
    {
      lout<<"step="<<*timestep<<", time=" << (*physical_time)
      << ", N="<< simulation_stats->m_particle_count << ", Kin.E="<<simulation_stats->m_kinetic_energy
      <<", vel=["<<simulation_stats->m_min_vel<<";"<<simulation_stats->m_max_vel<<"]"
      <<", acc=["<<simulation_stats->m_min_acc<<";"<<simulation_stats->m_max_acc<<"]" <<std::endl;
    }

  };
    
  // === register factories ===  
  ONIKA_AUTORUN_INIT(print_simulation_stats)
  {
   OperatorNodeFactory::instance()->register_factory( "print_simulation_stats", make_simple_operator<PrintSimulationStatistics> );
  }

}

