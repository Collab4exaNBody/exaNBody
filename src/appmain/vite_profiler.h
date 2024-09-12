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

#include <string>
#include <functional>
#include <iostream>
#include <chrono>

#include <onika/omp/ompt_task_timing.h>

#include <onika/trace/trace_output_format.h>
#include "app_config.h"

using ViteEventColor = onika::RGBColord;

struct ViteTraceElement
{
  const void* app_ctx = nullptr;
  const char* tag = nullptr;
  ssize_t rsc_id = -1;
  std::chrono::nanoseconds start;
  std::chrono::nanoseconds end;
  const std::string task_label() const;
};

using ViteFilterFunction   = std::function< bool          (const ViteTraceElement& e) >; 
using ViteLabelFunction    = std::function< std::string   (const ViteTraceElement& e) >;
using ViteColoringFunction = std::function< ViteEventColor(const ViteTraceElement& e) >; 

// standard per tag color customization function
extern ViteFilterFunction g_vite_default_filter;
extern ViteColoringFunction g_vite_tag_rnd_color;

// profiling trace
extern std::chrono::nanoseconds g_vite_trace_start_time;
extern std::chrono::nanoseconds g_vite_trace_max_duration;
extern std::chrono::nanoseconds g_vite_trace_min_duration;

void vite_start_trace(
  const xsv2ConfigStruct_trace& config ,
  onika::trace::TraceOutputFormat* formatter ,
  const ViteLabelFunction& label,
  const ViteColoringFunction& color,
  const ViteFilterFunction& filter = g_vite_default_filter
  );

void vite_process_event(const onika::omp::OpenMPToolTaskTiming& e);
void vite_end_trace(const xsv2ConfigStruct_trace& config );

