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
#include <memory>

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/domain.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <onika/memory/allocator.h>
#include <exanb/grid_cell_particles/grid_cell_values.h>
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <onika/string_utils.h>

#include <mpi.h>
#include <cstdio>

namespace exanb
{

  template<class GridT>
  class WriteGridVTKLegacy : public OperatorNode
  {
    ADD_SLOT( MPI_Comm       , mpi                 , INPUT , REQUIRED );
    ADD_SLOT( GridT       , grid       , INPUT , REQUIRED );
    ADD_SLOT( Domain         , domain       , INPUT , REQUIRED );
    ADD_SLOT( std::string    , filename , INPUT , REQUIRED );
    ADD_SLOT( GridCellValues , grid_cell_values , INPUT , REQUIRED );

  public:

    // -----------------------------------------------
    // -----------------------------------------------
    inline void execute ()  override final
    {
//      static constexpr size_t scalar_len = 18;
//      static const char* real_format = "% .10e\n";  
      
      const IJK dims = grid->dimension();
      const ssize_t gl = grid->ghost_layers();      
      const IJK dom_dims = domain->grid_dimension();
      const size_t dom_n_cells = dom_dims.i*dom_dims.j*dom_dims.k;
      const Vec3d dom_origin = domain->origin();
      const double cell_size = domain->cell_size();

      int rank=0, np=1;
      MPI_Comm_rank(*mpi, &rank);
      MPI_Comm_size(*mpi, &np);

      ldbg << "writing to "<< *filename << std::endl;
      MPI_File mpiofile;
      MPI_File_open(*mpi,filename->c_str(), MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL,&mpiofile);

      size_t global_offset = 0;
      ssize_t all_subdiv = -1;
      bool field_data_initialized = false;

      for( const auto& fp : grid_cell_values->fields() )
      {
        auto field_data = grid_cell_values->field_data( fp.second );
        ssize_t subdiv = fp.second.m_subdiv;
        ssize_t n_subcells = subdiv*subdiv*subdiv;
        assert( fp.second.m_components % n_subcells == 0 );
        ssize_t comps_per_subcell = fp.second.m_components / n_subcells;
        const double sub_cellsize = cell_size / subdiv;

        if( all_subdiv!=-1 && all_subdiv!=subdiv)
        {
          lerr << "insconsistent subdivision for field "<<fp.first<<" "<<subdiv<<" different from previous subdivision "<<all_subdiv <<std::endl;
          std::abort();
        }
        all_subdiv = subdiv;
                
        long header_size = 0;
        if( rank == 0 )
        {
          std::ostringstream oss;
          if( global_offset == 0 )
          {
            oss << "# vtk DataFile Version 2.0\n"
                << "ExaStamp V2 Grid Cell Values\n"
                << "BINARY\n"
                << "DATASET STRUCTURED_POINTS\n"
                << "DIMENSIONS "<<dom_dims.i*subdiv<<" "<<dom_dims.j*subdiv<<" "<<dom_dims.k*subdiv<<"\n"
                << "ORIGIN "<<dom_origin.x<<" "<<dom_origin.y<<" "<<dom_origin.z<<"\n"
                << "SPACING "<<sub_cellsize<<" "<<sub_cellsize<<" "<<sub_cellsize<<"\n"
                << "POINT_DATA "<<dom_n_cells*n_subcells<<"\n"
                << "SCALARS "<<fp.first<<" double "<<comps_per_subcell<<"\n"
                << "LOOKUP_TABLE default\n";
          }
          else
          {
            oss << "\n";
            if( ! field_data_initialized )
            {
              oss << "FIELD additional "<<grid_cell_values->fields().size()-1<<"\n";
              field_data_initialized = true;
            }
            oss <<fp.first<<" "<<comps_per_subcell<<" "<<dom_n_cells*n_subcells<<" double\n";
          }
              
          std::string header = oss.str();
          header_size = header.length();
          mpio_write( mpiofile, header.c_str(), global_offset, header_size );
        }

        // share field start offsets with all processors
        MPI_Bcast(&header_size,1,MPI_LONG,0,*mpi);
        global_offset += header_size;

        IJK dims_no_ghost = dims-2*gl;
        size_t n_inner_cells = dims_no_ghost.i * dims_no_ghost.j * dims_no_ghost.k;
        std::vector< double > scalar_data( n_inner_cells * n_subcells * comps_per_subcell );
        std::vector< std::pair<ssize_t,size_t> > global_order( n_inner_cells * n_subcells , {-1,0} );

#       pragma omp parallel
        {
          GRID_OMP_FOR_BEGIN(dims_no_ghost,_,loc)
          {
            const IJK cell_loc = loc + gl;
            const size_t cell_i = grid_ijk_to_index( dims , cell_loc );
            
            for(int ck=0;ck<subdiv;ck++)
            for(int cj=0;cj<subdiv;cj++)
            for(int ci=0;ci<subdiv;ci++)
            {
              IJK sc { ci, cj, ck };
              size_t j = cell_i * field_data.m_stride +  grid_ijk_to_index( IJK{subdiv,subdiv,subdiv} , sc ) * comps_per_subcell;
              
              size_t local_index = grid_ijk_to_index(dims_no_ghost*subdiv,loc*subdiv+sc);
              size_t global_index = grid_ijk_to_index( dom_dims*subdiv , ( cell_loc + grid->offset() ) * subdiv + sc );
              
              // write data to local buffer
              //char* buf_ptr = scalar_data[local_index].data();
              //std::snprintf( buf_ptr, scalar_len, real_format, field_data.m_data_ptr[j] );
              //buf_ptr[ scalar_len-1 ] = '\n';
              for(int c=0;c<comps_per_subcell;c++)
              {
                double x = field_data.m_data_ptr[ j + c ];
                SwapEnd(x);
                scalar_data[ local_index * comps_per_subcell + c ] = x;
              }

              global_order[ local_index ] = { global_index , local_index };            
            }
          }
          GRID_OMP_FOR_END
        }

        // allocate space for ordered local buffer
        std::vector< double > scalar_data_ordered( n_inner_cells * n_subcells * comps_per_subcell );

        // copy to ordered buffer
  #     pragma omp parallel for
        for(size_t i=0; i<(n_inner_cells * n_subcells); i++)
        {
          for(int c=0;c<comps_per_subcell;c++)
          {
            scalar_data_ordered[i*comps_per_subcell + c] = scalar_data[ global_order[i].second * comps_per_subcell + c ];
          }
        }

        // free unordered buffer
        scalar_data.clear();

        // execute coalesced writes
        if( global_order.size() > 0 )
        {
          ssize_t write_at = global_order[0].first;
          ssize_t read_at = 0;
          ssize_t count = 1;
          for(size_t i=1;i<(n_inner_cells * n_subcells); i++)
          {
            ssize_t go = global_order[i].first;
            if( go == (write_at+count) ) { ++count; }
            else
            {
              // flush
              mpio_write( mpiofile, scalar_data_ordered.data()+read_at*comps_per_subcell , global_offset+write_at*comps_per_subcell*sizeof(double) , count*comps_per_subcell*sizeof(double) );
              // start new
              read_at = i;
              write_at = go;
              count = 1;
            }
          }
          // final flush
          mpio_write( mpiofile, scalar_data_ordered.data()+read_at*comps_per_subcell , global_offset+write_at*comps_per_subcell*sizeof(double) , count*comps_per_subcell*sizeof(double) );
        }

        global_offset += dom_n_cells * n_subcells * comps_per_subcell * sizeof(double);
      }
 
      MPI_File_close(&mpiofile);
    }

    // -----------------------------------------------
    // -----------------------------------------------
    inline std::string documentation() const override final
    {
      return R"EOF(
Write a rectilinear grid's scalar fields to a legacy vtk file
)EOF";
    }

  private:
    static inline void mpio_write(MPI_File mpiofile, const void* buf, size_t at, size_t n)
    {
      MPI_Status status;
      MPI_File_write_at(mpiofile,at,buf,n,MPI_CHAR,&status);
      int write_count = 0;
      MPI_Get_count(&status, MPI_CHAR, &write_count);
      if( static_cast<size_t>(write_count) != n )
      {
        lerr << "WriteGridCellValues: write error, write_count="<<write_count<<", requested="<<n << std::endl;
        std::abort();
      }
    }

    // Convert bigEndian littleEndian
    // Thanks to https://stackoverflow.com/questions/105252
    template <typename T>
    void SwapEnd(T& var)
    {
      char* varArray = reinterpret_cast<char*>(&var);
      for(long i = 0; i < static_cast<long>(sizeof(var)/2); i++)
        std::swap(varArray[sizeof(var) - 1 - i],varArray[i]);
    }
  
  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(write_grid_vtklegacy)
  {
   OperatorNodeFactory::instance()->register_factory("write_grid_vtklegacy", make_grid_variant_operator< WriteGridVTKLegacy > );
  }

}
