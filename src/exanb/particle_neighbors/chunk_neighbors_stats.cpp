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
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/particle_id_codec.h>
#include <onika/memory/allocator.h>

#include <exanb/particle_neighbors/chunk_neighbors.h>
#include <exanb/particle_neighbors/chunk_neighbors_iterator.h>

#include <iomanip>
#include <mpi.h>

namespace exanb
{
  

  template<typename GridT>
  class ChunkNeighborsStats : public OperatorNode
  {
    ADD_SLOT( MPI_Comm           , mpi             , INPUT , MPI_COMM_WORLD  );
    ADD_SLOT( GridT              , grid            , INPUT , REQUIRED );
    ADD_SLOT( GridChunkNeighbors , chunk_neighbors , INPUT , GridChunkNeighbors{} );
    ADD_SLOT( double             , nbh_dist        , INPUT , 0.0 );
    ADD_SLOT( bool               , ghost           , INPUT , false );
    ADD_SLOT( double             , min_pair_dist   , INPUT , 0.0 );
    ADD_SLOT( double             , max_pair_dist   , INPUT , OPTIONAL );

  public:
    inline void execute () override final
    {
      if( !grid.has_value() ) { return; }
      if( grid->number_of_particles()==0 ) { return; }
        
      size_t cs = chunk_neighbors->m_chunk_size;
      size_t cs_log2 = 0;
      while( cs > 1 )
      {
        assert( (cs&1)==0 );
        cs = cs >> 1;
        ++ cs_log2;
      }
      cs = chunk_neighbors->m_chunk_size;
      ldbg << "cs="<<cs<<", log2(cs)="<<cs_log2<<std::endl;

      IJK dims = grid->dimension();
      unsigned long long total_nbh = 0;
      unsigned long long total_nbh_d2 = 0;
      //size_t total_nbh_chunk = 0;
      unsigned long long total_nbh_cells = 0;
      unsigned long long total_particles = grid->number_of_particles();
      
      unsigned long long nbh_d2_min = total_particles;
      unsigned long long nbh_d2_max = 0;

      unsigned long long nbh_min = total_particles;
      unsigned long long nbh_max = 0;
      total_particles = 0;

      auto cells = grid->cells();
      using CellT = std::remove_cv_t< std::remove_reference_t< decltype(cells[0]) > >;
      ChunkParticleNeighborsIterator<CellT> chunk_nbh_it_in = { grid->cells() , chunk_neighbors->data() , dims , chunk_neighbors->m_chunk_size };

      double check_min_dist = *min_pair_dist;
      double check_max_dist = std::numeric_limits<double>::max();
      if( max_pair_dist.has_value() ) check_max_dist = *max_pair_dist;

      const double nbh_d2 = (*nbh_dist) * (*nbh_dist) ;
      double nbh_min_dist = std::numeric_limits<double>::max();
      double nbh_max_dist = 0.;

#     pragma omp parallel
      {
        auto chunk_nbh_it = chunk_nbh_it_in;

        GRID_OMP_FOR_BEGIN(dims,cell_a,loc_a, schedule(dynamic) \
                                              reduction(+:total_particles,total_nbh,total_nbh_d2,total_nbh_cells) \
                                              reduction(min:nbh_d2_min,nbh_min,nbh_min_dist) \
                                              reduction(max:nbh_d2_max,nbh_max,nbh_max_dist) )
        {
          // std::cout<<"dims="<<dims<<" cell_a="<<cell_a<<" loc_a="<<loc_a<<std::endl;
          size_t n_particles_a = cells[cell_a].size();
          const double* __restrict__ rx_a = cells[cell_a][field::rx];
          const double* __restrict__ ry_a = cells[cell_a][field::ry];
          const double* __restrict__ rz_a = cells[cell_a][field::rz];

          const double* __restrict__ rx_b = nullptr; 
          const double* __restrict__ ry_b = nullptr;
          const double* __restrict__ rz_b = nullptr;

          // decode compacted chunks
          chunk_nbh_it.start_cell( cell_a , n_particles_a );
          for(size_t p_a=0;p_a<n_particles_a;p_a++)
          {
            if( (*ghost) ||  ! grid->is_ghost_cell(loc_a) )
            {
              ++ total_particles;
            }
            //size_t total_nbh_before = total_nbh;
            unsigned long long p_a_nbh_d2 = 0;
            unsigned long long p_a_nbh = 0;
            chunk_nbh_it.start_particle( p_a );
            size_t last_cell = std::numeric_limits<size_t>::max();
            while( ! chunk_nbh_it.end() )
            {
              size_t cell_b=0, p_b=0;
              chunk_nbh_it.get_nbh( cell_b , p_b );
              //std::cout<<"C"<<cell_a<<"P"<<p_a<<" -> C"<<cell_b<<"P"<<p_b<<std::endl;
              if( cell_b != last_cell )
              {
                rx_b = cells[cell_b][field::rx];
                ry_b = cells[cell_b][field::ry];
                rz_b = cells[cell_b][field::rz];
                last_cell = cell_b;
                if( (*ghost) || ! grid->is_ghost_cell(loc_a) )
                {
                  ++ total_nbh_cells;
                }
              }
              
              const double dx = rx_b[p_b] - rx_a[p_a];
              const double dy = ry_b[p_b] - ry_a[p_a];
              const double dz = rz_b[p_b] - rz_a[p_a];
              const double d2 = dx*dx+dy*dy+dz*dz;
              const double dist = sqrt(d2);
              if( dist < check_min_dist || dist > check_max_dist )
              {
                fatal_error() << "Bad particle pair distance detected : d="<<dist<<std::endl;
              }
              nbh_min_dist = std::min( nbh_min_dist , dist );
              nbh_max_dist = std::max( nbh_max_dist , dist );
              ++ p_a_nbh;
              if( d2 <= nbh_d2 ) { ++ p_a_nbh_d2; }
              
              chunk_nbh_it.next();
            }
                        
            if( (*ghost) ||  ! grid->is_ghost_cell(loc_a) )
            {
              total_nbh_d2 += p_a_nbh_d2;
              total_nbh += p_a_nbh;

              nbh_d2_min = std::min( nbh_d2_min , p_a_nbh_d2 );
              nbh_d2_max = std::max( nbh_d2_max , p_a_nbh_d2 );

              nbh_min = std::min( nbh_min , p_a_nbh );
              nbh_max = std::max( nbh_min , p_a_nbh );
            }
          }
                    
        }
        GRID_OMP_FOR_END
      }

      MPI_Allreduce(MPI_IN_PLACE,&total_particles,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,*mpi);
      MPI_Allreduce(MPI_IN_PLACE,&total_nbh,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,*mpi);
      MPI_Allreduce(MPI_IN_PLACE,&total_nbh_d2,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,*mpi);
      MPI_Allreduce(MPI_IN_PLACE,&total_nbh_cells,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,*mpi);

      MPI_Allreduce(MPI_IN_PLACE,&nbh_d2_min,1,MPI_UNSIGNED_LONG_LONG,MPI_MIN,*mpi);
      MPI_Allreduce(MPI_IN_PLACE,&nbh_min,1,MPI_UNSIGNED_LONG_LONG,MPI_MIN,*mpi);
      MPI_Allreduce(MPI_IN_PLACE,&nbh_min_dist,1,MPI_DOUBLE,MPI_MIN,*mpi);

      MPI_Allreduce(MPI_IN_PLACE,&nbh_d2_max,1,MPI_UNSIGNED_LONG_LONG,MPI_MAX,*mpi);
      MPI_Allreduce(MPI_IN_PLACE,&nbh_max,1,MPI_UNSIGNED_LONG_LONG,MPI_MAX,*mpi);
      MPI_Allreduce(MPI_IN_PLACE,&nbh_max_dist,1,MPI_DOUBLE,MPI_MAX,*mpi);

      lout << "===== Chunk Neighbors stats =====" << std::fixed << std::setprecision(2) << std::endl;
	    lout << "Chunk size             = "<<cs <<std::endl;
      lout << "Particles              = "<<total_particles<<std::endl;
      if(total_particles>0) lout << "Nbh cells (tot./avg)   = "<< total_nbh_cells <<" / "<< (total_nbh_cells*1.0/total_particles) <<std::endl;
      lout << "Neighbors (chunk/<d)   = "<<total_nbh <<" / "<<total_nbh_d2 << std::endl;
	    if(total_nbh>0) lout << "<d / chunk ratio       = " << (total_nbh_d2*100/total_nbh)*0.01 << " , storage eff. = "<< (total_nbh_d2*1.0/total_nbh)*cs <<std::endl;
      if(total_particles>0) lout << "Avg nbh (chunk/<d)     = "<< (total_nbh*1.0/total_particles) <<" / "<< (total_nbh_d2*1.0/total_particles) <<std::endl;
      lout << "min [chunk;<d] / Max [chunk;<d] = ["<< nbh_min<<";"<<nbh_d2_min <<"] / ["<< nbh_max <<";"<<nbh_d2_max<<"]"  <<std::endl;
	    if(total_nbh>0) lout << "distance : min / max   = " << nbh_min_dist << " / "<< nbh_max_dist << std::endl;
      lout << "=================================" << std::defaultfloat << std::endl;
    }
  
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(chunk_neighbors_stats)
  {
   OperatorNodeFactory::instance()->register_factory("chunk_neighbors_stats", make_grid_variant_operator< ChunkNeighborsStats > );
  }

}

