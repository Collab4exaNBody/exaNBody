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

#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/mpi/flat_buffer_ghosts.h>

#include <mpi.h>

namespace exanb
{

  template<class GridT, bool ToGhost>
  class FlatBufferGhostsNode : public OperatorNode
  {
    ADD_SLOT( MPI_Comm                 , mpi               , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( GhostCommunicationScheme , ghost_comm_scheme , INPUT , OPTIONAL );
    ADD_SLOT( GridT                    , grid              , INPUT_OUTPUT );
    ADD_SLOT( GhostFlatBuffer          , flat_buffer       , INPUT , REQUIRED , DocString{"per-particle rows in grid flat order (rebind it to the producer's buffer)"} );
    ADD_SLOT( long                     , mpi_tag           , INPUT , 211 , DocString{"MPI tag of the messages"} );

  public:
    inline std::string documentation() const override final
    {
      if constexpr ( ToGhost ) return R"EOF(
Copies the rows of a per-particle flat buffer (GhostFlatBuffer, grid flat order: cell particle offset + index in
cell) from owner particles to their ghost images, following the ghost communication scheme of the last ghost update.
Meant for wide per-particle data owned by a potential between two passes (e.g. message-passing networks). The slot
flat_buffer is usually rebound to the producer's output:

  grace_msg_forward:
    rebind: { flat_buffer: grace_message }
    body: [ ghost_update_flat_buffer ]
)EOF";
      else return R"EOF(
Sums the rows of a per-particle flat buffer (GhostFlatBuffer, grid flat order) of ghost particles into the rows of
their owners, following the ghost communication scheme of the last ghost update (ghost rows are left unchanged).
The reverse of ghost_update_flat_buffer, e.g. for gradients of per-particle messages:

  grace_msg_reverse:
    rebind: { flat_buffer: grace_message_grad }
    body: [ update_flat_buffer_from_ghost ]
)EOF";
    }

    inline void execute() override final
    {
      if( !ghost_comm_scheme.has_value() ) return;
      if( flat_buffer->m_data == nullptr || flat_buffer->m_row_size == 0 ) return;
      if( flat_buffer->m_row_stride < flat_buffer->m_row_size ) fatal_error() << "flat buffer row stride smaller than its row size" << std::endl;
      grid->rebuild_particle_offsets();
      if( flat_buffer->m_nrows < grid->number_of_particles() )
      {
        fatal_error() << "flat buffer has " << flat_buffer->m_nrows << " rows for " << grid->number_of_particles() << " particles (ghosts included)" << std::endl;
      }
      flat_buffer_ghost_exchange<ToGhost>( *mpi , *ghost_comm_scheme , grid->cell_particle_offset_data() , *flat_buffer , int(*mpi_tag) );
    }
  };

  template<class GridT> using GhostUpdateFlatBuffer = FlatBufferGhostsNode<GridT,true>;
  template<class GridT> using UpdateFlatBufferFromGhost = FlatBufferGhostsNode<GridT,false>;

  ONIKA_AUTORUN_INIT(update_flat_buffer_ghosts)
  {
    OperatorNodeFactory::instance()->register_factory( "ghost_update_flat_buffer", make_grid_variant_operator<GhostUpdateFlatBuffer> );
    OperatorNodeFactory::instance()->register_factory( "update_flat_buffer_from_ghost", make_grid_variant_operator<UpdateFlatBufferFromGhost> );
  }

}
