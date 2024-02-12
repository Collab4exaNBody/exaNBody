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
#pragma once

#include <fstream>
#include <onika/color_scale.h>
//#include "vite_event_color.h"

namespace onika
{
  namespace trace
  {

    class TraceOutputFormat
    {
    public:
      inline std::ostream& stream() { return m_out; }
      
      virtual ~TraceOutputFormat() = default;
      
      virtual void open(const std::string& fname);
      virtual void close();

      virtual void start_trace() {}
      virtual void declare_state(const std::string& idname, const std::string& fullname, const RGBColord& col) {}
      virtual void declare_threads(int n, double start, double end) {}
      virtual void set_state(int t, const std::string& idname, double timepoint, double duration=-1.0, double max_duration=-1.0 ) {}
      virtual void finalize_trace(int n, double time) {}
      virtual void add_idle_plot( const std::vector<double>& values , double start, double end);
      virtual void add_total_time( const std::vector< std::pair<double,std::string> >& total_times );
      
    private:
      std::string m_filename;
      std::ofstream m_out;
    };

  }
  
}

