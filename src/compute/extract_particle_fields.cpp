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
#include <exanb/core/particle_field_bundle.h>

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstring>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace exanb
{

  template<class GridT>
  struct ExtractParticleFields : public OperatorNode
  {
    ADD_SLOT( MPI_Comm                 , mpi            , INPUT  , MPI_COMM_WORLD );
    ADD_SLOT( GridT                    , grid           , INPUT  , REQUIRED );
    ADD_SLOT( std::vector<std::string> , fields         , INPUT  , std::vector<std::string>{} , DocString{"Names of the fields to copy (e.g. [rx, vx, virial]). Empty means every field that can be exported. 'id' is always included when the grid has it."} );
    ADD_SLOT( bool                     , include_ghosts , INPUT  , false , DocString{"If true, ghost-layer particles are included"} );
    ADD_SLOT( bool                     , gather         , INPUT  , false , DocString{"If true, particles of all MPI processes are gathered on rank 0 (other ranks get empty arrays), in rank order. Otherwise each process copies its own particles."} );
    ADD_SLOT( ParticleFieldBundle      , particles      , OUTPUT , DocString{"Copied fields, one column per field in native type, all in the same cell-major particle order"} );

    // sink: keep the node even though no other operator consumes its output
    inline bool is_sink() const override final { return true; }

    inline void execute() override final
    {
      const std::set<std::string> requested( fields->begin() , fields->end() );

      std::vector<std::string> available;
      collect_names( typename GridT::Fields{} , available );
      for(const auto& f : requested)
      {
        if( std::find(available.begin(),available.end(),f) == available.end() )
        {
          std::ostringstream oss;
          for(const auto& a : available) oss << " " << a;
          fatal_error() << "extract_particle_fields: unknown field '" << f << "', available fields are:" << oss.str() << std::endl;
        }
      }

      particles->columns.clear();
      particles->count = 0;
      copy_fields( typename GridT::Fields{} , requested );
      if( *gather ) gather_on_root();
    }

    inline std::string documentation() const override final
    {
      return
        "Copy per-particle fields from the grid into a ParticleFieldBundle, readable from Python "
        "as a dict of numpy arrays (Vec3d -> (N,3), Mat3d -> (N,3,3), Quaternion -> (N,4) as w,x,y,z). "
        "Particles are emitted in cell-major (k,j,i) order; ghost particles are excluded by default.";
    }

  private:
    // concatenate every column of all ranks on rank 0, in rank order
    inline void gather_on_root()
    {
      MPI_Comm comm = *mpi;
      int rank = 0, nprocs = 1;
      MPI_Comm_rank( comm , &rank );
      MPI_Comm_size( comm , &nprocs );
      if( nprocs == 1 ) return;

      unsigned long long local_count = particles->count, total_count = 0;
      MPI_Reduce( &local_count , &total_count , 1 , MPI_UNSIGNED_LONG_LONG , MPI_SUM , 0 , comm );

      // every rank holds the same columns (same grid type, same requested fields)
      for( auto& [name,col] : particles->columns )
      {
        if( col.data.size() > size_t(INT_MAX) ) fatal_error() << "extract_particle_fields: column '" << name << "' too large to gather" << std::endl;
        int local_bytes = static_cast<int>( col.data.size() );
        std::vector<int> bytes( rank==0 ? nprocs : 0 ) , displ( rank==0 ? nprocs : 0 );
        MPI_Gather( &local_bytes , 1 , MPI_INT , bytes.data() , 1 , MPI_INT , 0 , comm );
        std::vector<uint8_t> all;
        if( rank == 0 )
        {
          size_t total_bytes = 0;
          for(int p=0;p<nprocs;p++)
          {
            if( total_bytes > size_t(INT_MAX) ) fatal_error() << "extract_particle_fields: column '" << name << "' too large to gather" << std::endl;
            displ[p] = static_cast<int>( total_bytes );
            total_bytes += bytes[p];
          }
          all.resize( total_bytes );
        }
        MPI_Gatherv( col.data.data() , local_bytes , MPI_BYTE , all.data() , bytes.data() , displ.data() , MPI_BYTE , 0 , comm );
        col.data = std::move( all );
      }
      particles->count = ( rank == 0 ) ? total_count : 0;
    }

    template<class... fid>
    static inline void collect_names( FieldSet<fid...> , std::vector<std::string>& names )
    {
      ( ... , names.push_back( onika::soatl::FieldId<fid>{}.short_name() ) );
    }

    template<class... fid>
    inline void copy_fields( FieldSet<fid...> , const std::set<std::string>& requested )
    {
      ( ... , copy_field<fid>( requested ) );
    }

    template<class FID>
    inline void copy_field( const std::set<std::string>& requested )
    {
      using value_type = typename onika::soatl::FieldId<FID>::value_type;
      using Layout = ParticleFieldLayout<value_type>;

      onika::soatl::FieldId<FID> fobj{};
      const std::string name = fobj.short_name();
      const bool wanted = requested.empty() || requested.count(name) > 0 || name == "id";
      if( ! wanted ) return;

      if constexpr ( ! Layout::supported )
      {
        // silently skipped when copying every field, an error when asked for explicitly
        if( ! requested.empty() ) fatal_error() << "extract_particle_fields: field '" << name << "' has a type that cannot be exported" << std::endl;
      }
      else
      {
        using scalar_type = typename Layout::scalar_type;
        static_assert( sizeof(value_type) % sizeof(scalar_type) == 0 , "field value type is not a packed array of scalars" );

        auto cells = grid->cells();
        const IJK dims = grid->dimension();
        const ssize_t gl = *include_ghosts ? 0 : ssize_t( grid->ghost_layers() );
        const IJK inner = dims - (2 * gl);

        ParticleFieldColumn& col = particles->columns[name];
        col.dtype = particle_field_dtype<scalar_type>();
        col.shape = Layout::shape();
        col.data.clear();
        col.data.reserve( grid->number_of_particles() * sizeof(value_type) );

        size_t count = 0;
        for( ssize_t k = 0; k < inner.k; ++k )
        for( ssize_t j = 0; j < inner.j; ++j )
        for( ssize_t i = 0; i < inner.i; ++i )
        {
          const ssize_t ci = grid_ijk_to_index( dims , IJK{ i + gl, j + gl, k + gl } );
          const size_t n = cells[ci].size();
          const uint8_t* ptr = reinterpret_cast<const uint8_t*>( cells[ci][fobj] );
          col.data.insert( col.data.end() , ptr , ptr + n * sizeof(value_type) );
          count += n;
        }
        particles->count = count;
      }
    }
  };

  template<class GridT> using ExtractParticleFieldsTmpl = ExtractParticleFields<GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(extract_particle_fields)
  {
    OperatorNodeFactory::instance()->register_factory( "extract_particle_fields" , make_grid_variant_operator<ExtractParticleFieldsTmpl> );
  }

}
