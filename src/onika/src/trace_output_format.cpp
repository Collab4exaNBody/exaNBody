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
#include <onika/trace/trace_output_format.h>

#include <fstream>
#include <iostream>

namespace onika
{
  namespace trace
  {

    void TraceOutputFormat::open(const std::string& fname)
    {
      m_filename=fname;
      m_out.open(fname);
    }

    void TraceOutputFormat::close()
    {
      m_out.close();
    }

    void TraceOutputFormat::add_idle_plot( const std::vector<double>& values , double start, double end )
    {
      std::string fname = m_filename + ".idle";
      std::ofstream out( fname );
      if( ! out.good() )
      {
        std::cerr << "can't open file '" << fname << "'" << std::endl;
        return;
      }
      //std::cout << "write '"<<fname<<"' "<<values.size()<<" values, range=["<<start<<";"<<end<<"]"<< std::endl;
      size_t n = values.size();
      for(size_t i=0;i<n;i++)
      {
        out << start+(i*(end-start)/(n-1))  << " " << values[i] << std::endl;
      }

      std::string pfname = m_filename + ".plot";
      std::ofstream pout( pfname );
      if( ! pout.good() )
      {
        std::cerr << "can't open file '" << pfname << "'" << std::endl;
        return;
      }
      pout << "set xrange ["<<start<<":"<<end<<"]\n";
      pout << "plot \""<<fname<<"\" using 1:2 with lines\n";
    }

    void TraceOutputFormat::add_total_time( const std::vector< std::pair<double,std::string> >& total_times )
    {
      std::string fname = m_filename + ".total";
      std::ofstream out( fname );
      if( ! out.good() )
      {
        std::cerr << "can't open file '" << fname << "'" << std::endl;
        return;
      }
      std::cout << "write '"<<fname<<"' ("<<total_times.size()<<" entries)" << std::endl;
      for(const auto& x:total_times)
      {
        out << x.second << " " << x.first << std::endl;
      }
    }

  }
}


