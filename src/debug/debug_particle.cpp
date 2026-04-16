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
#include <onika/log.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <onika/math/basic_types_stream.h>
#include <onika/math/quaternion_stream.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <onika/print_utils.h>
#include <exanb/core/print_particle.h>
#include <onika/soatl/field_tuple.h>

#include <vector>
#include <sstream>
#include <set>
#include <numeric>

#include <mpi.h>

namespace exanb
{

  // ================== Thermodynamic state compute operator ======================

  template<typename GridT
    , class = AssertGridHasFields< GridT, field::_id >
    >
  class DebugParticleNode : public OperatorNode
  {
    using ParticleIds = std::vector<uint64_t>;

    ADD_SLOT( MPI_Comm , mpi    , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( GridT       , grid      , INPUT, REQUIRED);
    ADD_SLOT( ParticleIds , ids       , INPUT, ParticleIds{} );
    ADD_SLOT( bool        , ghost     , INPUT, false );
    ADD_SLOT( std::string , filename  , INPUT, OPTIONAL );

  public:
  
    inline void execute () override final
    {
      ParticleIds sids = *ids;
      std::sort( sids.begin(), sids.end() );

      auto cells = grid->cells();
      IJK dims = grid->dimension();
      
      static constexpr size_t MAX_STR_LEN = 1024;
      std::string local_debug_lines;
      local_debug_lines.reserve( sids.size() * MAX_STR_LEN );
      
#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN( dims, i, loc )
        {
          const uint64_t* __restrict__ part_ids = cells[i][field::id];
          bool is_ghost_cell = grid->is_ghost_cell( loc );
          size_t n_part = cells[i].size();
          for(size_t j=0;j<n_part;j++)
          {
            if( ( sids.empty() || std::binary_search( sids.begin(), sids.end(), part_ids[j] ) ) && ( (*ghost) || !is_ghost_cell ) )
            {
              std::ostringstream oss;
              oss << onika::default_stream_format << std::defaultfloat << std::scientific << std::setprecision(6)
                  << std::setfill('0') << std::setw(12) << part_ids[j] << ' ' << ( is_ghost_cell ? 'G' : 'O' ) <<" @"<<( loc + grid->offset() ) <<" ";
              print_particle( oss , cells[i][j] );
              std::string s = oss.str();
#             pragma omp critical
              {
                size_t cur_pos = local_debug_lines.size();
                local_debug_lines.resize( local_debug_lines.size() + MAX_STR_LEN , ' ' );
                std::strncpy(local_debug_lines.data()+cur_pos,s.c_str(),MAX_STR_LEN);
                local_debug_lines[cur_pos+MAX_STR_LEN-1]='\0';
              }
            }
          }
        }
        GRID_OMP_FOR_END
      }

      int nprocs = 1;
      int rank = 0;
      MPI_Comm_size(*mpi,&nprocs);
      MPI_Comm_rank(*mpi,&rank);

      std::vector<int> data_counts( nprocs , 0 );
      data_counts[rank] = local_debug_lines.size();
      MPI_Allreduce(MPI_IN_PLACE,data_counts.data(),nprocs,MPI_INT,MPI_SUM,*mpi);
      std::vector<int> data_displs( nprocs , 0 );
      std::exclusive_scan( data_counts.begin() , data_counts.end() , data_displs.begin() , 0 );
      size_t total_size = 0;
      for(const auto x:data_counts) total_size += x;
      assert( total_size % MAX_STR_LEN == 0 );
      std::string all_debug_lines( total_size , ' ' );
      MPI_Gatherv( local_debug_lines.data() , local_debug_lines.size() , MPI_CHAR , all_debug_lines.data() , data_counts.data() , data_displs.data() , MPI_CHAR , 0 , *mpi );

      if( rank == 0 )
      {
        const size_t n_lines = total_size / MAX_STR_LEN;
        std::set< std::string > all_sorted_lines;
        for(size_t i=0;i<n_lines;i++)
        {
          const char* s = all_debug_lines.data() + i * MAX_STR_LEN;
          all_sorted_lines.insert( s );
        }
        
        bool to_file = false;
        std::ofstream fout;
        if( filename.has_value() )
        {
          fout.open(*filename);
          to_file = true;
        }
        if( to_file ) ldbg << "debug particles to file "<< (*filename) << std::endl;
        for(const auto & l : all_sorted_lines)
        {
          if( ! l.empty() && l[0]!=' ' )
          {
            if(to_file) fout << l;
            else lout << l;
          }
        }
        if( to_file ) fout.close();
      }
    }

  };
  
  template<class GridT> using DebugParticleNodeTmpl = DebugParticleNode<GridT>;
  
  // === register factories ===
  ONIKA_AUTORUN_INIT(debug_particle)
  {
   OperatorNodeFactory::instance()->register_factory( "debug_particle", make_grid_variant_operator<DebugParticleNodeTmpl> );
  }

}

