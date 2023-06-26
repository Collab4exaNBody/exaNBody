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


