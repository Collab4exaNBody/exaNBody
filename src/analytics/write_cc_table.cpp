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
    
  public:

    // -----------------------------------------------
    // -----------------------------------------------
    inline void execute ()  override final
    {
      ldbg << "CC table size = "<< cc_table->size() << std::endl;

      const char* csv_header = "label ; count ; center_x ; center_y ; center_z\n";
      const size_t csv_header_size = std::strlen(csv_header);
      const char* csv_sample_format = "% 012llu ; % 012llu ; % .9e ; % .9e ; % .9e \n";
      const size_t csv_sample_size = std::snprintf( nullptr, 0, csv_sample_format, 0ull, 0ull, 0., 0., 0. );
      const std::string csv_filename = (*filename) + "_table.csv";
      ldbg <<"write_cc_table : header = "<<csv_header ;
      ldbg <<"write_cc_table : format = "<<csv_sample_format ;
      ldbg <<"write_cc_table : samples="<<cc_table->size()<<" , header_size="<<csv_header_size<<" , sample_size="<<csv_sample_size <<"' , file='"<< csv_filename <<"'"<< std::endl;
      
      MpioIOText outfile = { csv_header       , csv_header_size ,
                             csv_sample_format, csv_sample_size };

      outfile.open( *mpi , csv_filename );

      for(size_t i=0;i<cc_table->size();i++)
      {
        const unsigned long long cell_label = static_cast<ssize_t>(cc_table->at(i).m_label);
        const unsigned long long global_id = cc_table->at(i).m_rank;
        const unsigned long long cell_count = cc_table->at(i).m_cell_count;
        const double cx = cc_table->at(i).m_center.x;
        const double cy = cc_table->at(i).m_center.y;
        const double cz = cc_table->at(i).m_center.z;
        outfile.write_sample( global_id , cell_label, cell_count, cx, cy, cz );
      }
      
      outfile.close();
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
  ONIKA_AUTORUN_INIT(write_cc_table)
  {
   OperatorNodeFactory::instance()->register_factory("write_cc_table", make_simple_operator< CCTableCSVWriter > );
  }

/*
# Example Programmable filter in Paraview to use CSV table as a lookup table to convert cc_label to volume or other property
*/

}
