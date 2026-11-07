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

#include <onika/log.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/grid.h>
#include <exanb/core/grid_algorithm.h>
#include <exanb/core/grid_fields.h>
#include <exanb/core/make_grid_variant_operator.h>

#include <string>
#include <vector>

namespace exanb
{

  template<class GridT>
  struct ExtractParticleField : public OperatorNode
  {
    ADD_SLOT( GridT             , grid          , INPUT  , REQUIRED                                        );
    ADD_SLOT( std::string       , field_name    , INPUT  , "rx"  , DocString{"Name of the field to copy (e.g. rx, vy, id, type)"} );
    ADD_SLOT( bool              , include_ghosts, INPUT  , false , DocString{"If true, ghost-layer particles are included"} );
    ADD_SLOT( std::vector<double>, data         , OUTPUT ,         DocString{"Flattened per-particle values cast to double, cell-major order"} );

    // Mark this operator as a sink so the graph optimiser never suppresses it.
    // Without this, onika's post_graph_build() would remove any terminal node
    // whose output slots are not consumed by another operator in the graph.
    inline bool is_sink() const override final { return true; }

    inline void execute() override final
    {
      data->clear();
      data->reserve( grid->number_of_particles() );

      bool found = false;
      copy_matching_field( typename GridT::Fields{}, found );

      if ( !found )
        lerr << "extract_particle_field: unknown field '" << *field_name
             << "' for this grid type" << std::endl;
    }

    // Iterate over every field id in the GridT field set and copy the one
    // whose short_name() matches field_name.
    template<class... fid>
    inline void copy_matching_field( FieldSet<fid...>, bool& found )
    {
      ( ... , try_copy<fid>( found ) );
    }

    template<class FID>
    inline void try_copy( bool& found )
    {
      if ( found ) return;

      onika::soatl::FieldId<FID> fobj{};
      if ( std::string(fobj.short_name()) != *field_name ) return;
      found = true;

      using value_type = typename onika::soatl::FieldId<FID>::value_type;

      auto cells      = grid->cells();
      IJK  dims       = grid->dimension();
      ssize_t gl      = *include_ghosts ? 0 : ssize_t( grid->ghost_layers() );
      IJK  inner      = dims - (2 * gl);

      for ( ssize_t k = 0; k < inner.k; ++k )
      for ( ssize_t j = 0; j < inner.j; ++j )
      for ( ssize_t i = 0; i < inner.i; ++i )
      {
        IJK loc { i + gl, j + gl, k + gl };
        ssize_t ci = grid_ijk_to_index( dims, loc );
        size_t   n = cells[ci].size();
        const value_type* ptr = cells[ci][fobj];
        for ( size_t p = 0; p < n; ++p )
          data->push_back( static_cast<double>( ptr[p] ) );
      }
    }

    inline std::string documentation() const override final
    {
      return
        "Copy a named per-particle field from the grid into a contiguous "
        "std::vector<double> output slot.  The vector is accessible from "
        "Python via slot_as_array().  Particles are emitted in cell-major "
        "(k,j,i) order; ghost particles are excluded by default.";
    }
  };

  template<class GridT> using ExtractParticleFieldTmpl = ExtractParticleField<GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(extract_particle_field)
  {
    OperatorNodeFactory::instance()->register_factory(
      "extract_particle_field",
      make_grid_variant_operator<ExtractParticleFieldTmpl> );
  }

} // namespace exanb
