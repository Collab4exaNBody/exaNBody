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

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/domain.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <onika/memory/allocator.h>
#include <exanb/grid_cell_particles/grid_cell_values.h>
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <onika/string_utils.h>

#include <mpi.h>
#include <cstdio>
#include <experimental/filesystem>

namespace exanb
{

  template<class GridT>
  class WriteGridVTK : public OperatorNode
  {
    ADD_SLOT( MPI_Comm       , mpi              , INPUT , REQUIRED );
    ADD_SLOT( GridT       , grid             , INPUT , REQUIRED );
    ADD_SLOT( Domain         , domain           , INPUT , REQUIRED );
    ADD_SLOT( std::string    , filename         , INPUT , "grid" );
    ADD_SLOT( GridCellValues , grid_cell_values , INPUT , REQUIRED );
    ADD_SLOT( bool           , use_point_data   , INPUT , true );

  public:

    // -----------------------------------------------
    // -----------------------------------------------
    inline void execute ()  override final
    {
      namespace fs = std::experimental::filesystem;
    
      
#     ifndef NDEBUG
      static constexpr size_t scalar_len = 18;
      static const char* real_format = "% .10e\n";  
      const std::string test_str = onika::format_string(real_format,123456.0);
      const size_t scalar_len_test = test_str.length();
      assert( scalar_len == scalar_len_test );
      assert( test_str[scalar_len-1] == '\n' );
#     endif

      const IJK dims = grid->dimension();
      const ssize_t gl = grid->ghost_layers();      
      const IJK dom_dims = domain->grid_dimension();
      //const size_t dom_n_cells = dom_dims.i*dom_dims.j*dom_dims.k;
      //const Vec3d dom_origin = domain->origin();
      //const double cell_size = domain->cell_size();
      const IJK dims_no_ghost = dims-2*gl;

      int rank=0, np=1;
      MPI_Comm_rank(*mpi, &rank);
      MPI_Comm_size(*mpi, &np);

      ssize_t subdiv = -1;
      for(const auto& fp:grid_cell_values->fields())
      {
        if( subdiv == -1 ) subdiv = fp.second.m_subdiv;
        else if( subdiv != ssize_t(fp.second.m_subdiv) )
        {
          lerr << "inconsistent grid subdivision accross scalar fields" << std::endl;
          std::abort();
        }
      }
      
      if( subdiv < 1 )
      {
        return;
      }

      GridBlock local_block = enlarge_block( grid->block() , -gl );
      std::vector<GridBlock> all_blocks( np );
      MPI_Allgather( (char*) &local_block, sizeof(GridBlock), MPI_CHAR, (char*) all_blocks.data(), sizeof(GridBlock), MPI_CHAR, *mpi);
      assert( all_blocks[rank] == local_block);
      
      bool point_mode = *use_point_data;

      const IJK whole_ext = (point_mode)? dom_dims*subdiv - 1 : dom_dims*subdiv;
      
      ldbg << "extent="<<whole_ext<<", point_mode="<<std::boolalpha<<point_mode <<std::endl;

      std::string basename = *filename;
      if( basename.rfind('.') != std::string::npos )
      {
        basename = basename.substr(0,basename.rfind('.'));
      }
      if( rank == 0 ) 
      {
        ldbg << "create directory basename"<<std::endl;
        fs::remove_all(basename);
        std::error_code ec;
        fs::create_directory(basename, ec);
      }
      MPI_Barrier(*mpi);

      if( rank == 0 ) 
      {
        std::string pvti_filename = *filename + ".pvti" ;
        ldbg << "write file "<<pvti_filename<<" ..."<<std::endl;
        std::ofstream pvti( pvti_filename );
        pvti << "<VTKFile type=\"PImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
        pvti << "  <PImageData WholeExtent=\"0 "
             <<whole_ext.i<<" 0 "
             <<whole_ext.j<<" 0 "
             <<whole_ext.k<<"\" GhostLevel=\"0\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n";
        bool scalars_set = true;
        for(const auto& fp:grid_cell_values->fields())
        {
          size_t side = fp.second.m_subdiv;
          size_t n_subcells = side*side*side;
          size_t n_comps = fp.second.m_components / n_subcells;
          assert( n_comps*n_subcells == fp.second.m_components );
          if( scalars_set )
          {
            if( point_mode ) pvti << "    <PPointData Scalars=\""<<fp.first<<"\">\n";
            else             pvti << "    <PCellData Scalars=\""<<fp.first<<"\">\n";
            scalars_set = false;
          }
          pvti << "      <PDataArray type=\"Float64\" NumberOfComponents=\""<<n_comps<<"\" Name=\"" << fp.first << "\"/>\n";
        }
        if( point_mode ) pvti <<"    </PPointData>\n";
        else             pvti <<"    </PCellData>\n";
        for(int i=0;i<np;i++)
        {
          IJK start, end;
          end.i = ((all_blocks[i].end.i*subdiv > whole_ext.i) && point_mode) ? all_blocks[i].end.i*subdiv-1 : all_blocks[i].end.i*subdiv;
          end.j = ((all_blocks[i].end.j*subdiv > whole_ext.j) && point_mode) ? all_blocks[i].end.j*subdiv-1 : all_blocks[i].end.j*subdiv; 
          end.k = ((all_blocks[i].end.k*subdiv > whole_ext.k) && point_mode) ? all_blocks[i].end.k*subdiv-1 : all_blocks[i].end.k*subdiv; 
          
          if(point_mode)
            {
              start.i = (all_blocks[i].start.i*subdiv > 0) ? all_blocks[i].start.i*subdiv-1 : 0; 
              start.j = (all_blocks[i].start.j*subdiv > 0) ? all_blocks[i].start.j*subdiv-1 : 0; 
              start.k = (all_blocks[i].start.k*subdiv > 0) ? all_blocks[i].start.k*subdiv-1 : 0; 
            }
          else
            {
              start.i = all_blocks[i].start.i*subdiv; 
              start.j = all_blocks[i].start.j*subdiv; 
              start.k = all_blocks[i].start.k*subdiv; 
            }
          pvti << "    <Piece Extent=\""
               << start.i<<" "<<end.i<<" "
               << start.j<<" "<<end.j<<" "
               << start.k<<" "<<end.k
               <<"\" Source=\""<<basename<<"/"<<"piece"<<i<<".vti\"/>\n";
        }
        pvti <<"  </PImageData>\n";
        pvti <<"</VTKFile>\n";
      }


      // write local processor's piece .vti file
      std::ostringstream vti_filename_oss;
      vti_filename_oss <<basename<<"/"<<"piece"<<rank<<".vti";
      std::string vti_filename = vti_filename_oss.str();
      ldbg << "write file "<<vti_filename<<" ..."<<std::endl;
      std::ofstream vti( vti_filename );

      // compute local piece dimensions and data array block size
      const IJK local_subgrid_dims = dimension( local_block ) * subdiv;
      uint64_t block_elements = grid_cell_count(local_subgrid_dims);
      const uint64_t block_size = block_elements * sizeof(double);

      IJK dims_copy = dims_no_ghost;
      IJK sg_dims = dims_no_ghost*subdiv;
      
      IJK start_loc_ext, end_loc_ext;
      end_loc_ext.i = ((local_block.end.i*subdiv > whole_ext.i) && point_mode) ? local_block.end.i*subdiv-1 : local_block.end.i*subdiv;
      end_loc_ext.j = ((local_block.end.j*subdiv > whole_ext.j) && point_mode) ? local_block.end.j*subdiv-1 : local_block.end.j*subdiv;
      end_loc_ext.k = ((local_block.end.k*subdiv > whole_ext.k) && point_mode) ? local_block.end.k*subdiv-1 : local_block.end.k*subdiv;
      
      if (point_mode)
        {
          start_loc_ext.i = (local_block.start.i*subdiv > 0) ? local_block.start.i*subdiv-1 : 0;
          start_loc_ext.j = (local_block.start.j*subdiv > 0) ? local_block.start.j*subdiv-1 : 0;
          start_loc_ext.k = (local_block.start.k*subdiv > 0) ? local_block.start.k*subdiv-1 : 0;
        }
      else
        {
          start_loc_ext.i = local_block.start.i*subdiv;
          start_loc_ext.j = local_block.start.j*subdiv;
          start_loc_ext.k = local_block.start.k*subdiv;
        }

      vti<<"<VTKFile type=\"ImageData\" version=\"2.2\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
      vti<<"  <ImageData WholeExtent=\""
         << start_loc_ext.i <<" "<< end_loc_ext.i <<" "
         << start_loc_ext.j <<" "<< end_loc_ext.j <<" "
         << start_loc_ext.k <<" "<< end_loc_ext.k <<"\" Origin=\"0 0 0\" Spacing=\"1 1 1\" Direction=\"1 0 0 0 1 0 0 0 1\">\n";
      vti<<"    <Piece Extent=\"";
      if(point_mode)
        {
          vti<<
            local_block.start.i*subdiv<<" "<< local_block.end.i*subdiv-1 <<" "<<
            local_block.start.j*subdiv<<" "<< local_block.end.j*subdiv-1 <<" "<<
            local_block.start.k*subdiv<<" "<< local_block.end.k*subdiv-1 <<"\">\n";
        }
      else
        {
          vti<<
            local_block.start.i*subdiv<<" "<< local_block.end.i*subdiv <<" "<<
            local_block.start.j*subdiv<<" "<< local_block.end.j*subdiv <<" "<<
            local_block.start.k*subdiv<<" "<< local_block.end.k*subdiv <<"\">\n";
        }
      bool scalars_set = true;
      size_t offset = 0;
      for(const auto& fp:grid_cell_values->fields())
      {
        size_t side = fp.second.m_subdiv;
        size_t n_subcells = side*side*side;
        size_t n_comps = fp.second.m_components / n_subcells;
        assert( n_comps*n_subcells == fp.second.m_components );
        uint64_t vec_block_size = block_size * n_comps;
        if( scalars_set )
        {
          if( point_mode ) vti << "      <PointData Scalars=\""<<fp.first<<"\">\n";
          else             vti << "      <CellData Scalars=\""<<fp.first<<"\">\n";
          scalars_set = false;
        }
        vti << "        <DataArray type=\"Float64\" NumberOfComponents=\""<<n_comps<<"\" Name=\"" << fp.first << "\" format=\"appended\" offset=\""<< offset <<"\"/>\n";
        offset += sizeof(uint64_t) + vec_block_size;
      }
      if( point_mode ) vti<<"      </PointData>\n";
      else             vti<<"      </CellData>\n";
      vti<<"    </Piece>\n";
      vti<<"  </ImageData>\n";
      vti<<"  <AppendedData encoding=\"raw\">\n";
      vti<<"    _";

      assert( dims_no_ghost*subdiv == local_subgrid_dims );

      offset = 0;
      std::vector<double> buffer;
      for(const auto& fp:grid_cell_values->fields())
      {
        size_t side = fp.second.m_subdiv;
        size_t n_subcells = side*side*side;
        size_t n_comps = fp.second.m_components / n_subcells;
        uint64_t vec_block_size = block_size * n_comps;
        size_t vec_block_elements = block_elements * n_comps;
        
        assert( n_comps*n_subcells == fp.second.m_components );
        ldbg << "\twrite field "<<fp.first<<std::endl;
        
        // write block size
        vti.write( reinterpret_cast<const char*>(&vec_block_size), sizeof(uint64_t) );
        
        buffer.assign( vec_block_elements , 0.0 );
        auto field_data = grid_cell_values->field_data( fp.second );

        GRID_FOR_BEGIN(dims_copy,_,loc)
        {
          const IJK cell_loc = loc + gl; 
          const size_t cell_i = grid_ijk_to_index( dims , cell_loc );
          for(int ck=0;ck<subdiv;ck++)
          for(int cj=0;cj<subdiv;cj++)
          for(int ci=0;ci<subdiv;ci++)
          {
            const IJK sc { ci, cj, ck }; 
            IJK sg_loc = loc*subdiv+sc; 
            assert( sg_loc.i>=0 && sg_loc.j>=0 && sg_loc.k>=0 );
            assert( sg_loc.i < sg_dims.i && sg_loc.j < sg_dims.j && sg_loc.k < sg_dims.k );
            size_t gcv_index = cell_i * field_data.m_stride +  grid_ijk_to_index( IJK{subdiv,subdiv,subdiv} , sc ) * n_comps;             
            size_t buffer_index = grid_ijk_to_index(sg_dims,sg_loc) * n_comps;
            for(unsigned int c=0;c<n_comps;c++)
              {
                buffer[ buffer_index + c ] = field_data.m_data_ptr[ gcv_index + c ];
              }
          }
        }
        GRID_FOR_END
        
        vti.write( reinterpret_cast<const char*>(buffer.data()), vec_block_size );
        offset += sizeof(uint64_t) + vec_block_size;
      }
      vti<<"\n  </AppendedData>\n</VTKFile>\n";
    }

    // -----------------------------------------------
    // -----------------------------------------------
    inline std::string documentation() const override final
    {
      return R"EOF(
Write a rectilinear grid's scalar fields to a .pvti vtk file (along with its .vti sub files)
)EOF";
    }
  
  };

  // === register factories ===
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory("write_grid_vtk", make_grid_variant_operator< WriteGridVTK > );
  }

}
