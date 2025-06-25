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
#include <exanb/core/grid_algorithm.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <onika/log.h>
#include <exanb/core/domain.h>
#include <onika/string_utils.h>
#include <exanb/amr/amr_grid.h>
#include <exanb/amr/amr_grid_algorithm.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/grid_cell_particles/grid_cell_values.h>

#include <mpi.h>
#include <onika/mpi/all_reduce_multi.h>

namespace exanb
{
  
  
  // ================== Thermodynamic state compute operator ======================

  template<class GridT>
  struct GridStats : public OperatorNode
  {  
    ADD_SLOT( MPI_Comm , mpi    , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( GridT    , grid   , INPUT , REQUIRED );   
    ADD_SLOT( Domain   , domain , INPUT , OPTIONAL );
    ADD_SLOT( AmrGrid  , amr    , INPUT , OPTIONAL );

    ADD_SLOT( GridCellValues , grid_cell_values , INPUT, OPTIONAL );

    inline void execute () override final
    {
      using onika::mpi::all_reduce_multi;
      
      IJK dims = grid->dimension();
      size_t ghost_layers = grid->ghost_layers();
      //size_t n_cells = grid.number_of_cells();
      
      ssize_t n_inner_cells = 0;
      ssize_t n_inner_particles = 0;
      ssize_t n_ghost_cells = 0;
      ssize_t n_ghost_particles = 0;
      ssize_t n_empty_ghost_cells = 0;
      ssize_t n_gpu_addressable = 0;
      ssize_t n_empty_cells = 0;
      
      ssize_t min_cell_particles = grid->number_of_particles();
      ssize_t max_cell_particles = 0;
      
      ssize_t min_cell_res = 256;
      ssize_t max_cell_res = 0;
      
      ssize_t n_oc = 0; // number of particles (geometrically outside) of their owning cell's volume
      ssize_t n_oc_ghost = 0; // same, but conting only those in ghost cells

      // AMR stats
      const size_t* sub_grid_start = amr.has_value() ?  amr->sub_grid_start().data() : nullptr;
      ssize_t n_subcells = 0;
      ssize_t total_cell_res = 0;

#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN(dims,i,loc, \
            schedule(dynamic) \
            reduction(+:total_cell_res,n_subcells,n_oc,n_oc_ghost,n_ghost_cells,n_ghost_particles,n_inner_cells,n_inner_particles,n_empty_cells,n_empty_ghost_cells,n_gpu_addressable) \
            reduction(min:min_cell_particles,min_cell_res) \
            reduction(max:max_cell_particles,max_cell_res) )
        {          
          const ssize_t n_part = grid->cell_number_of_particles(i);
          const bool is_ghost_cell = grid->is_ghost_cell(loc);

          if( grid->cell_is_gpu_addressable(i) ) ++ n_gpu_addressable;

          for(ssize_t j=0;j<n_part;j++)
          {
            if( ! is_inside_exclude_upper( grid->cell_bounds(loc) , grid->particle_position(i,j) ) )
            {
              ++ n_oc;
              if( is_ghost_cell ) ++ n_oc_ghost;
            }
          }
          if( is_ghost_cell )
          {
            ++ n_ghost_cells;
            n_ghost_particles += n_part;
            if(n_part==0) ++n_empty_ghost_cells;
          }
          else
          {
            ++ n_inner_cells;
            n_inner_particles += n_part;
            if(n_part==0) ++n_empty_cells;
            if( sub_grid_start != nullptr )
            {
              size_t sc = sub_grid_start[i+1] - sub_grid_start[i] + 1;
              n_subcells += sc;
              ssize_t cell_res = icbrt64( sc );
              min_cell_res = std::min( min_cell_res , cell_res );
              max_cell_res = std::max( max_cell_res , cell_res );
              total_cell_res += cell_res;
            }
            min_cell_particles = std::min( min_cell_particles , n_part );
            max_cell_particles = std::max( max_cell_particles , n_part );
          }
       }
       GRID_OMP_FOR_END
     }

     min_cell_particles            = -min_cell_particles;
     min_cell_res                  = -min_cell_res;
     long long min_inner_particles = -n_inner_particles;
     long long min_ghost_particles = -n_ghost_particles;
     long long min_total_particles = -( n_inner_particles + n_ghost_particles );
     long long max_inner_particles = n_inner_particles;
     long long max_ghost_particles = n_ghost_particles;
     long long max_total_particles = n_inner_particles + n_ghost_particles;

     all_reduce_multi(*mpi,MPI_SUM,ssize_t{}, total_cell_res,n_subcells,n_oc,n_oc_ghost,n_ghost_cells,n_ghost_particles,n_inner_cells,n_inner_particles,n_empty_cells,n_empty_ghost_cells,n_gpu_addressable );

     all_reduce_multi(*mpi,MPI_MAX,ssize_t{}, min_cell_particles, max_cell_particles, min_cell_particles, max_cell_res
                     ,min_inner_particles,min_ghost_particles,min_total_particles,max_inner_particles,max_ghost_particles,max_total_particles);
     min_cell_particles  = -min_cell_particles;
     min_cell_res        = -min_cell_res;
     min_inner_particles = -min_inner_particles;
     min_ghost_particles = -min_ghost_particles;
     min_total_particles = -min_total_particles;
     
     ssize_t icdiv = n_inner_cells; if(icdiv==0) icdiv=1;
     lout << "========== grid stats ===========" << std::endl;
     double cell_size = grid->cell_size();
     dims = grid->dimension();
     if( domain.has_value() )
     {
       dims = domain->grid_dimension();     
       assert( cell_size == domain->cell_size() );
       cell_size = domain->cell_size();
     }
     lout << "dimension      = " << dims.i <<'x'<< dims.j<<'x'<< dims.k << std::endl;
     lout << "ghost layers   = " << ghost_layers << std::endl;
     lout << "cell size      = " << cell_size << std::endl;
     lout << "inner cells    = " << onika::large_integer_to_string(n_inner_cells) << std::endl;
     lout << "ghost cells    = " << onika::large_integer_to_string(n_ghost_cells) << std::endl;
     lout << "empty cells    = " << onika::large_integer_to_string(n_empty_cells) << " (" <<onika::large_integer_to_string(n_empty_ghost_cells)<<" ghosts)" << std::endl;
     lout << "part. per cell = " << onika::large_integer_to_string(min_cell_particles) <<" / "
                                 << onika::large_integer_to_string(n_inner_particles / icdiv) << " / "
                                 << onika::large_integer_to_string(max_cell_particles) << std::endl;
     lout << "inner part.    = " << onika::large_integer_to_string(n_inner_particles) << " , min "
                                 << onika::large_integer_to_string(min_inner_particles) <<" , max "
                                 << onika::large_integer_to_string(max_inner_particles) << std::endl;
     lout << "ghost part.    = " << onika::large_integer_to_string(n_ghost_particles) << " , min "
                                 << onika::large_integer_to_string(min_ghost_particles) <<" , max "
                                 << onika::large_integer_to_string(max_ghost_particles) << std::endl;
     lout << "total part.    = " << onika::large_integer_to_string(n_inner_particles+n_ghost_particles) << " , min "
                                 << onika::large_integer_to_string(min_total_particles) << " , max "
                                 << onika::large_integer_to_string(max_total_particles) << std::endl;
     lout << "otb particles  = " << onika::large_integer_to_string(n_oc);
     if( n_oc_ghost > 0 ) lout << " ("<<onika::large_integer_to_string(n_oc_ghost)<<" in ghost cells)";
     lout <<std::endl;
     
     lout << "GPU addr cells = " << onika::large_integer_to_string(n_gpu_addressable) <<std::endl;
     lout << "AMR info avail = " << std::boolalpha << (sub_grid_start != nullptr) << std::endl;
     if( (n_inner_cells-n_empty_cells) > 0 )
     {
	     lout << "AMR res        = " << onika::format_string("%lld / %.2g / %lld",min_cell_res,total_cell_res*1.0/(n_inner_cells-n_empty_cells),max_cell_res) <<std::endl;
	   }
	   if( n_subcells > 0 )
	   {
       lout << "AMR density    = " << onika::format_string("%.2g",n_inner_particles*1.0/n_subcells) <<std::endl;
     }
     if( grid_cell_values.has_value() )
     {
       lout << "Cell values    = " << grid_cell_values->m_fields.size() << std::endl;
       for(const auto& p : grid_cell_values->m_fields)
       {
         lout << "  " << p.first << " : subdiv="<< p.second.m_subdiv<<", components="<<p.second.m_components<<std::endl;
       }
     }
     lout << "PFA Calculator = ";
     onika::soatl::pfa_size_calculator_t< GridT::Alignment , typename GridT::field_set_t , GridT::ChunkSize >::print( lout );
     lout << "=================================" << std::endl;
    }
  };
    
  // === register factories ===  
  ONIKA_AUTORUN_INIT(grid_stats)
  {
    const auto strvec = xnb_grid_variants_as_strings();
    ldbg << "Available grid variants :";
    for(const auto& s:strvec) ldbg << " " << s;
    ldbg << std::endl;
  
    OperatorNodeFactory::instance()->register_factory( "grid_stats", make_grid_variant_operator< GridStats > );
  }

}

