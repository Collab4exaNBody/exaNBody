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
    ADD_SLOT( MPI_Comm                , mpi     , INPUT , MPI_COMM_WORLD , DocString{"MPI communicator"} );
    ADD_SLOT( Domain                  , domain   , INPUT , OPTIONAL , DocString{"MPI communicator"} );
    ADD_SLOT( ConnectedComponentTable , cc_table , INPUT , REQUIRED );
    ADD_SLOT( std::string             , filename , INPUT , "cc" );
    ADD_SLOT( bool                    , write_stats , INPUT , false );
    
  public:

    // -----------------------------------------------
    // -----------------------------------------------
    inline void execute ()  override final
    {
      ldbg << "CC table size = "<< cc_table->size() << std::endl;

      const char* csv_header = "label ; rank ; count ; center_x ; center_y ; center_z\n";
      const size_t csv_header_size = std::strlen(csv_header);
      const char* csv_sample_format = "% 012llu ; % 06llu ; % 012llu ; % .9e ; % .9e ; % .9e \n";
      const size_t csv_sample_size = std::snprintf( nullptr, 0, csv_sample_format, 0ull, 0ull, 0ull, 0., 0., 0. );
      const std::string csv_filename = (*filename) + "_table.csv";
      ldbg <<"write_cc_table : header = "<<csv_header ;
      ldbg <<"write_cc_table : format = "<<csv_sample_format ;
      ldbg <<"write_cc_table : samples="<<cc_table->size()<<" , header_size="<<csv_header_size<<" , sample_size="<<csv_sample_size <<"' , file='"<< csv_filename <<"'"<< std::endl;
      
      MpioIOText outfile = { csv_header       , csv_header_size ,
                             csv_sample_format, csv_sample_size };

      outfile.open( *mpi , csv_filename );
      
      unsigned long long total_cell_count = 0;
      for(size_t i=0;i<cc_table->size();i++)
      {
        const unsigned long long global_id = static_cast<ssize_t>(cc_table->at(i).m_label);
        const unsigned long long rank = cc_table->at(i).m_rank;
        const unsigned long long cell_count = cc_table->at(i).m_cell_count;
        total_cell_count += cell_count;
        const double cx = cc_table->at(i).m_center.x;
        const double cy = cc_table->at(i).m_center.y;
        const double cz = cc_table->at(i).m_center.z;
        outfile.write_sample( global_id, global_id, rank, cell_count, cx, cy, cz );
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
        int rank=0, nprocs=1;
        MPI_Comm_rank( *mpi , &rank );
        MPI_Comm_size( *mpi , &nprocs );
        assert( rank < nprocs );
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
