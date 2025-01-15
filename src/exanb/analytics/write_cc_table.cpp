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
#include <exanb/core/domain.h>
#include <exanb/core/string_utils.h>
#include <exanb/analytics/cc_info.h>
#include <mpi.h>
#include <fstream>

namespace exanb
{
  class CCTableCSVWriter : public OperatorNode
  {
    ADD_SLOT( MPI_Comm                , mpi     , INPUT , MPI_COMM_WORLD , DocString{"MPI communicator"} );
    ADD_SLOT( Domain                  , domain   , INPUT , OPTIONAL , DocString{"MPI communicator"} );
    ADD_SLOT( ConnectedComponentTable , cc_table , INPUT , REQUIRED );
    ADD_SLOT( std::string             , filename , INPUT , "cc" );
    
  public:

    // -----------------------------------------------
    // -----------------------------------------------
    inline void execute ()  override final
    {
      ldbg << "CC table size = "<< cc_table->size() << std::endl;
      for(size_t i=0;i<cc_table->size();i++)
      {
        ldbg <<"CC #"<<i<<" : label="<<cc_table->at(i).m_label<<" count="<<cc_table->at(i).m_cell_count<<" center="<<cc_table->at(i).m_center<<std::endl;
      }
      
      int rank = 0 , nprocs = 1;
      MPI_Comm_rank( *mpi , &rank );
      MPI_Comm_size( *mpi , &nprocs );

      std::string csv_filename = (*filename) + "_table.csv";
      MPI_File fileh;
      int rc = MPI_File_open( *mpi , csv_filename.c_str() , MPI_MODE_CREATE | MPI_MODE_RDWR , MPI_INFO_NULL , &fileh );
      if( rc != MPI_SUCCESS )
      {
        fatal_error() << "MpiIO: unable to open file '" << *filename << "' for writing" << std::endl << std::flush ;
      }

      static const std::string csv_header = "label ; global_id ; owner ; count ; center_x ; center_y ; center_z\n";
      static const std::string csv_format = "%09d ; %09d ; %06d ; %09d ; % .6e ; % .6e ; % .6e\n";
      static const std::string csv_sample = format_string( csv_format , 1 , 2 , 3 , 4, -3.5 , 56.36589845454 , -123 );
      static const size_t header_size = csv_header.length();
      static const size_t sample_size = csv_sample.length();

      if( rank == 0)
      {
        MPI_File_write_at( fileh , 0 , csv_header.c_str() , header_size , MPI_CHAR , MPI_STATUSES_IGNORE );
      }
      
      std::string cc_str;
      for(size_t i=0;i<cc_table->size();i++)
      {
        const ssize_t cell_label = static_cast<ssize_t>(cc_table->at(i).m_label);
        const ssize_t global_id = cc_table->at(i).m_rank;
        format_string_inplace( cc_str , csv_format , cell_label , global_id , rank , cc_table->at(i).m_cell_count , cc_table->at(i).m_center.x , cc_table->at(i).m_center.y , cc_table->at(i).m_center.z );
        assert( cc_str.length() == sample_size );
        MPI_File_write_at( fileh , header_size + ( global_id * sample_size ) , cc_str.c_str() , sample_size , MPI_CHAR , MPI_STATUSES_IGNORE );
        ldbg << cc_str ;
      }
      
      MPI_File_close( &fileh );
    }

    // -----------------------------------------------
    // -----------------------------------------------
    inline std::string documentation() const override final
    {
      return R"EOF(
Connected components table writer, outputs a CSV file containing CC properties
This file can be used from Paraview's programmable filter to transform cc_label property on a grid
to CC properties, such as volume (sub cell count), using a python script as follows :
# Paraview programmable filter script
import numpy as np
input0 = inputs[0]
cc_id = input0.PointData["cc_label"]
ts = 600010
fp = "."
cc_table = np.genfromtxt("%s/cc_%09d_table.csv" % (fp,ts) , dtype=None, names=True, delimiter=';', autostrip=True)
cc_count = cc_table["count"]
cc_label = input0.PointData["cc_label"]
output.PointData.append( cc_label , "cc_label" )
N = cc_label.size
cc_volume = np.array( [] , dtype=np.uint32 )
cc_volume.resize(N)
cc_volume.reshape([N])
for i in range(N):
    assert i < cc_volume.size , "index %d out of cc_volume bounds (%d)" % (i,cc_volume.size)
    label_idx = int(cc_label[i])
    if label_idx >= 0:
      assert label_idx < cc_count.size , "%d out of cc_count bounds (%d)" % (label_idx,cc_count.size)
      cc_volume[i] = cc_count[label_idx]
    else:
      cc_volume[i] = 0
output.PointData.append( cc_volume , "cc_volume" )
)EOF";
    }

  };

  // === register factories ===
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory("write_cc_table", make_simple_operator< CCTableCSVWriter > );
  }

/*
# Example Programmable filter in Paraview to use CSV table as a lookup table to convert cc_label to volume or other property
*/

}
