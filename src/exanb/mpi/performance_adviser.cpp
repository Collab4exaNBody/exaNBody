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
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/log.h>
#include <exanb/core/domain.h>
#include <onika/string_utils.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>

#include <mpi.h>
#include <omp.h>
#include <cmath>

namespace exanb
{
  

  template<class GridT>
  struct PerformanceAdviser : public OperatorNode
  {
    ADD_SLOT( MPI_Comm  , mpi        , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( Domain    , domain     , INPUT , REQUIRED );
    ADD_SLOT( GridT  , grid       , INPUT , REQUIRED );
    ADD_SLOT( bool      , verbose    , INPUT , false );
    ADD_SLOT( double    , ghost_dist , INPUT , 0.0 );  // thickness of ghost particle layer, in grid space.

    inline void execute () override final
    {
      static constexpr size_t min_particles_per_thread = 100;
      static constexpr size_t max_small_cells_per_thread = 100;
      static constexpr size_t max_cells_per_thread = 100000;

      int np=1, nt=1;      
//      int rank = 0;
//      MPI_Comm_rank(*mpi,&rank);
      MPI_Comm_size(*mpi,&np);
#     pragma omp parallel
      {
#       pragma omp single
        nt = omp_get_num_threads();
      }
      
      unsigned long n_particles = grid->number_of_particles();
      MPI_Allreduce(MPI_IN_PLACE,&n_particles,1,MPI_UNSIGNED_LONG,MPI_SUM,*mpi);
      
      size_t n_domain_cells = grid_cell_count(domain->grid_dimension()) ;
      // size_t avg_cells_per_mpi_proc = n_domain_cells / np;

      if( *verbose )
      {
        lout << "========= Performance ===========" << std::endl;
        lout << "Particles        = "<<large_integer_to_string(n_particles)<<std::endl;
        lout << "Cells            = "<<large_integer_to_string(n_domain_cells)<<std::endl;
//        lout << "Ghost layers     = "<< grid->ghost_layers() <<std::endl;        
        lout << "MPI Processes    = "<<large_integer_to_string(np)<<std::endl;
        lout << "OMP threads      = "<<large_integer_to_string(nt)<<std::endl;
        lout << "MPI x OMP        = "<<large_integer_to_string(np*nt)<<std::endl;
        lout << "Cells / MPI Proc = "<<large_integer_to_string(n_domain_cells/np)<<std::endl;        
        lout << "Cells / Thread   = "<<large_integer_to_string(n_domain_cells/(np*nt))<<std::endl;        
        lout << "Part. / MPI Proc = "<<large_integer_to_string(n_particles/np)<<std::endl;        
        lout << "Part. / Thread   = "<<large_integer_to_string(n_particles/(np*nt))<<std::endl;        
      }
      
      if( (n_domain_cells / (np*nt)) > max_small_cells_per_thread && domain->cell_size() <= (*ghost_dist) )
      {
        double adv_cell_size = std::ceil( (*ghost_dist) * 10.5 ) / 10.0;
        lout << "Alert       : too many ("<<(n_domain_cells/(np*nt))<<") small sized ("<<format_string("%.2f ang",domain->cell_size())<<") cells per thread" << std::endl;
        lout << "Tip         : consider increasing cell_size to "<< format_string("[ %.1f ang ; %.1f ang ]",adv_cell_size,adv_cell_size*8) << std::endl;
      }
      
      if( (n_domain_cells / (np*nt)) > max_cells_per_thread )
      {
        double mpi_count = (n_domain_cells / nt)*1.0 / max_cells_per_thread;
        size_t c=1; while(c<mpi_count) c*=2;
        lout << "Alert       : too many cells per thread ("<<(n_domain_cells/(np*nt))<<")" << std::endl;
        lout << "Tip         : consider increasing number of MPI processes to "<< c << std::endl;
      }
      
      if( n_particles/(np*nt) < min_particles_per_thread )
      {
        double mpi_count = std::max( 1.0 , std::ceil( (n_particles / nt)*1.0 / min_particles_per_thread ) );
        lout << "Alert       : low number of particles per thread ("<< n_particles/(np*nt) << ")" << std::endl;
        lout << "Tip         : consider decreasing number of MPI processes to "<< mpi_count << std::endl;
      }

      if( *verbose )
      {
        lout << "=================================" << std::endl << std::endl;
      }
    }
        
  };

  // === register factory ===
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "performance_adviser", make_grid_variant_operator<PerformanceAdviser> );
  }

}

