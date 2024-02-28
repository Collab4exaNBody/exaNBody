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

#include <exanb/particle_neighbors/chunk_neighbors.h>

#include <exanb/compute/compute_pair_buffer.h>
#include <exanb/compute/compute_pair_optional_args.h>
#include <exanb/compute/compute_pair_traits.h>

#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/field_sets.h>
#include <exanb/core/particle_id_codec.h>
#include <exanb/core/log.h>

#include <onika/soatl/field_id.h>
#include <onika/soatl/field_pointer_tuple.h>
#include <onika/cuda/cuda.h>
#include <onika/cuda/stl_adaptors.h>
#include <onika/integral_constant.h>
#include <onika/flat_tuple.h>

#include <cstddef>

namespace exanb
{
  using DefaultPositionFields = onika::FlatTuple< onika::soatl::FieldId<field::_rx> , onika::soatl::FieldId<field::_ry> , onika::soatl::FieldId<field::_rz> >;
  template<class... PFields> using PosititionFields = onika::FlatTuple< onika::soatl::FieldId<PFields> ... >;

  template< class ComputeBufferT, class CellsT, class FieldAccessorTupleT, size_t ... FieldIndex >
  ONIKA_HOST_DEVICE_FUNC
  static inline void compute_cell_particle_pairs_pack_nbh_fields( ComputeBufferT& tab, CellsT cells, size_t cell_b, size_t p_b, const FieldAccessorTupleT& nbh_fields , std::index_sequence<FieldIndex...> )
  {
    using NbhTuple = typename ComputeBufferT::NbhFieldTuple;
    tab.nbh_pt[tab.count] = NbhTuple { cells[cell_b][nbh_fields.get(onika::tuple_index_t<FieldIndex>{})][p_b] ... };
  }

}

