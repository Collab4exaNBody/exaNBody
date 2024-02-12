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

#include <onika/trace/trace_output_format.h>
#include <unordered_map>

namespace onika
{
  namespace trace
  {


    class YAMLTraceFormat : public TraceOutputFormat
    {
    public:
      void open(const std::string& fname) override final;
      void close() override final; 
      void start_trace() override final;
      void declare_threads(int n, double start, double end) override final;
      void declare_state(const std::string& idname, const std::string& fullname, const RGBColord& col) override final;
      void set_state(int t, const std::string& idname, double timepoint, double duration, double max_duration ) override final;
      void finalize_trace(int n, double time) override final;

      void add_idle_plot( const std::vector<double>& values , double start, double end) override final;
    };

  }
}


