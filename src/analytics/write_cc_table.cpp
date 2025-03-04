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
#include <exanb/core/domain.h>
#include <exanb/analytics/cc_info.h>
#include <exanb/io/mpi_file_io.h>
#include <mpi.h>
#include <fstream>

namespace exanb
{
  using onika::scg::OperatorNode;
  using onika::scg::OperatorNodeFactory;
  using onika::scg::make_simple_operator;

  class CCTableCSVWriter : public OperatorNode
  {
    using StringVector = std::vector< std::string >;

    ADD_SLOT( MPI_Comm                , mpi     , INPUT , MPI_COMM_WORLD , DocString{"MPI communicator"} );
    ADD_SLOT( Domain                  , domain   , INPUT , OPTIONAL , DocString{"MPI communicator"} );
    ADD_SLOT( ConnectedComponentTable , cc_table , INPUT , REQUIRED );
    ADD_SLOT( std::string             , filename , INPUT , "cc" );

    ADD_SLOT( StringVector            , write_custom_fields  , INPUT , StringVector{} ); 
    ADD_SLOT( bool                    , write_stats , INPUT , false );
    ADD_SLOT( bool                    , write_rank , INPUT , true );
    ADD_SLOT( bool                    , write_gyration , INPUT , true );
    
  public:

    // -----------------------------------------------
    // -----------------------------------------------
    inline void execute ()  override final
    {
      int rank=0, nprocs=1;
      MPI_Comm_rank( *mpi , &rank );
      MPI_Comm_size( *mpi , &nprocs );
      assert( rank < nprocs );

      ldbg << "CC table size = "<< cc_table->size() << std::endl;

      std::set<std::string> enabled_custom_fields;
      for(const auto & cf : *write_custom_fields )
      {
        auto it = std::find( cc_table->m_custom_field_name.begin() , cc_table->m_custom_field_name.end() , cf );
        if( it != cc_table->m_custom_field_name.end() )
        {
          enabled_custom_fields.insert( cf );
        }
      }

      std::string csv_header;
      {
        std::ostringstream header_oss;
        header_oss << "label ; count ; center_x ; center_y ; center_z";
        if( *write_rank ) header_oss << " ; rank";
        if( *write_gyration ) header_oss << " ; gyr_xx ; gyr_xy ; gyr_xz ; gyr_yx ; gyr_yy ; gyr_yz ; gyr_zx ; gyr_zy ; gyr_zz";
        for( size_t cfi=0 ; cfi < cc_table->m_custom_field_name.size() ; cfi++ )
        {
          if( enabled_custom_fields.find( cc_table->m_custom_field_name[cfi] ) != enabled_custom_fields.end() )
          {
            const size_t vecsize = cc_table->m_custom_field_vecsize[cfi];
            for(size_t vi=0;vi<vecsize;vi++)
            {
              header_oss << " ; " << cc_table->m_custom_field_name[cfi];
              if( vecsize > 1 ) header_oss <<"_"<<vi;
            }
          }
        }
        header_oss << "\n";
        csv_header = header_oss.str();
      }
      const size_t csv_header_size = csv_header.length();
      
      const char * csv_main_format = "% 012llu ; % 012llu ; % .9e ; % .9e ; % .9e";
      const char * csv_rank_format = " ; % 06llu";
      const char * csv_gyr_format = " ; % .9e ; % .9e ; % .9e ; % .9e ; % .9e ; % .9e ; % .9e ; % .9e ; % .9e";
      const char * csv_custom_format = " ; % .9e";

      const size_t csv_main_size = std::snprintf( nullptr, 0, csv_main_format, 0ull, 0ull, 0., 0., 0. );
      const size_t csv_rank_size = std::snprintf( nullptr, 0, csv_rank_format, 0ull );
      const size_t csv_gyr_size = std::snprintf( nullptr, 0, csv_gyr_format, 0., 0., 0., 0., 0., 0., 0., 0., 0. );
      const size_t csv_custom_size = std::snprintf( nullptr, 0, csv_custom_format, 0. );

      std::string csv_sample_format = csv_main_format;
      size_t csv_sample_size = csv_main_size;
      if( *write_rank )
      {
        csv_sample_format += csv_rank_format;
        csv_sample_size += csv_rank_size;
      }
      if( *write_gyration )
      {
        csv_sample_format += csv_gyr_format;
        csv_sample_size += csv_gyr_size;
      }
      for( size_t cfi=0 ; cfi < cc_table->m_custom_field_name.size() ; cfi++ )
      {
        if( enabled_custom_fields.find( cc_table->m_custom_field_name[cfi] ) != enabled_custom_fields.end() )
        {
          const size_t vecsize = cc_table->m_custom_field_vecsize[cfi];
          for(size_t vi=0;vi<vecsize;vi++)
          {
            csv_sample_format += csv_custom_format;
            csv_sample_size += csv_custom_size;
          }
        }
      }
      csv_sample_format += "\n";
      csv_sample_size ++;

      const std::string csv_filename = (*filename) + "_table.csv";

      ldbg <<"write_cc_table : header = "<<csv_header ;
      ldbg <<"write_cc_table : format = "<<csv_sample_format ;
      ldbg <<"write_cc_table : samples="<<cc_table->size()<<" , header_size="<<csv_header_size<<" , sample_size="<<csv_sample_size <<"' , file='"<< csv_filename <<"'"<< std::endl;
      
      MpioIOText outfile = { csv_header.c_str(), csv_header_size, csv_sample_format.c_str(), csv_sample_size };

      outfile.open( *mpi , csv_filename );
      
      std::vector<char> csv_line_buffer( csv_sample_size + 2 , '\0' );
      unsigned long long total_cell_count = 0;
      const size_t custom_field_values = cc_table->m_custom_field_values;
      for(size_t i=0;i<cc_table->size();i++)
      {
        const unsigned long long global_id = static_cast<ssize_t>(cc_table->at(i).m_label);
        const unsigned long long owner_rank = rank; //cc_table->at(i).m_rank;
        const unsigned long long cell_count = cc_table->at(i).m_cell_count;
        total_cell_count += cell_count;
        const auto& c = cc_table->at(i).m_center;
        const auto& g = cc_table->at(i).m_gyration;
        size_t buf_pos = 0;
        size_t n = std::snprintf( csv_line_buffer.data() + buf_pos , csv_main_size+1, csv_main_format, global_id, cell_count, c.x, c.y, c.z );
        assert( n == csv_main_size );
        buf_pos += n;
        if( *write_rank )
        {
          n = std::snprintf( csv_line_buffer.data() + buf_pos , csv_rank_size+1, csv_rank_format, owner_rank );
          assert( n == csv_rank_size );
          buf_pos += n;
        }
        if( *write_gyration )
        {
          n = std::snprintf( csv_line_buffer.data() + buf_pos , csv_gyr_size+1, csv_gyr_format, g.m11, g.m12, g.m13, g.m21, g.m22, g.m23, g.m31, g.m32, g.m33 );
          assert( n == csv_gyr_size );
          buf_pos += n;
        }
        for( size_t cfi=0 ; cfi < cc_table->m_custom_field_name.size() ; cfi++ )
        {
          if( enabled_custom_fields.find( cc_table->m_custom_field_name[cfi] ) != enabled_custom_fields.end() )
          {
            const size_t vecsize = cc_table->m_custom_field_vecsize[cfi];
            for(size_t vi=0;vi<vecsize;vi++)
            {
              const size_t value_index = custom_field_values * i + cc_table->m_custom_field_position[cfi] + vi;
              n = std::snprintf( csv_line_buffer.data() + buf_pos , csv_custom_size+1, csv_custom_format, cc_table->m_custom_field_data[value_index] );
              assert( n == csv_custom_size );
              buf_pos += n;              
            }
          }
        }
        csv_line_buffer[buf_pos++] = '\n';
        assert( buf_pos == csv_sample_size );
        outfile.write_sample_buf( global_id , csv_line_buffer.data() );
      }
      
      outfile.close();

      unsigned long long total_n_cc = cc_table->size();
      unsigned long long rank_cc_count_scan = 0;
      MPI_Exscan( &total_n_cc , &rank_cc_count_scan , 1 , MPI_UNSIGNED_LONG_LONG , MPI_SUM , *mpi );
      MPI_Allreduce( MPI_IN_PLACE , &total_n_cc , 1 , MPI_UNSIGNED_LONG_LONG , MPI_SUM , *mpi );
      MPI_Allreduce( MPI_IN_PLACE , &total_cell_count , 1 , MPI_UNSIGNED_LONG_LONG , MPI_SUM , *mpi );

      ldbg << "total_n_cc="<<total_n_cc<<" , total_cell_count="<<total_cell_count<<" , avg_cell_count="<<total_cell_count/total_n_cc<<" , rank_cc_count_scan="<<rank_cc_count_scan<<std::endl;

      if( *write_stats )
      {
        std::vector<unsigned long long> count_sum;
        if( rank == 0 ) count_sum.assign( nprocs , 0 );
        MPI_Gather(&rank_cc_count_scan, 1, MPI_UNSIGNED_LONG_LONG, count_sum.data(), 1, MPI_UNSIGNED_LONG_LONG, 0, *mpi);
        if( rank == 0 )
        {
          std::ofstream fout( (*filename)+"_stats.csv" );
          fout << "Ncc ; " << total_n_cc << std::endl;
          fout << "Tcount ; " << total_cell_count << std::endl;
          fout << "AVGcount ; " << std::setprecision(5) << total_cell_count*1.0/total_n_cc << std::endl;          
          fout << "Nmpi ; " << nprocs << std::endl;
          for(int i=0;i<nprocs;i++)
          {
            fout << "NccSum_"<<i<<" ; " << count_sum[i] << std::endl;
          }
          fout.flush();
          fout.close();
        }
      }
            
    }

    // -----------------------------------------------
    // -----------------------------------------------
    inline std::string documentation() const override final
    {
      return R"EOF(
Connected components table writer, outputs a CSV file containing CC properties
)EOF";
    }

  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(write_cc_table)
  {
   OperatorNodeFactory::instance()->register_factory("write_cc_table", make_simple_operator< CCTableCSVWriter > );
  }

/*
# Example Programmable filter in Paraview to use CSV table as a lookup table to convert cc_label to volume or other property
*/

}
