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
#include <exanb/core/grid.h>
#include <onika/math/basic_types.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/backup_r.h>

namespace exanb
{
  template<typename GridT>
  struct PositionBackupCheck : public OperatorNode
  {
    ADD_SLOT( GridT              , grid     , INPUT );
    ADD_SLOT( PositionBackupData , backup_r , INPUT);

    inline void execute ()  override final
    {
      const size_t grid_n_cells = grid->number_of_cells();
      const size_t backup_n_cells = backup_r->m_data.size();
      ldbg << "PositionBackupCheck : grid_n_cells="<<grid_n_cells<< " , backup_n_cells=" << backup_n_cells<< std::endl;
      if( grid_n_cells != backup_n_cells )
      {
        fatal_error() << "grid cells = "<<grid_n_cells<<" , backup_r cells = "<<backup_n_cells<<std::endl;
      }
    }

  };
    
 // === register factories ===  
  ONIKA_AUTORUN_INIT(backup_r)
  {
   OperatorNodeFactory::instance()->register_factory( "check_backup_r", make_grid_variant_operator< PositionBackupCheck > );
  }

}

