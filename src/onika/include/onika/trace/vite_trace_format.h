#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <onika/color_scale.h>
#include <onika/trace/trace_output_format.h>

namespace onika
{
  namespace trace
  {

    struct ViteOutputFunctions
    {
      using ViteHeaderFunction        = std::function< void (std::ostream& out) >;
      using ViteDelcareThreadFunction = std::function< void (std::ostream& out,int t,int n) >;
      using ViteDelcareStateFunction  = std::function< void (std::ostream& out,const std::string& idname, const std::string& fullname, const RGBColord& col ) >;
      using ViteSetStateFunction      = std::function< void (std::ostream& out,int t, const std::string& idname, double timepoint, double duration, double max_duration) >;
      using ViteEndFunction           = std::function< void (std::ostream& out,int n, double time) >;

      static const ViteHeaderFunction        g_vite_default_header;
      static const ViteDelcareThreadFunction g_vite_default_declare_thread;
      static const ViteDelcareStateFunction  g_vite_default_declare_state;
      static const ViteSetStateFunction      g_vite_default_set_state;
      //static const ViteSetStateFunction      g_vite_set_state_color_by_duration;
      static const ViteEndFunction           g_vite_default_finalize;

      ViteHeaderFunction header = g_vite_default_header;
      ViteDelcareThreadFunction declare_thread = g_vite_default_declare_thread;
      ViteDelcareStateFunction declare_state = g_vite_default_declare_state;
      ViteSetStateFunction set_state = g_vite_default_set_state;
      ViteEndFunction finalize = g_vite_default_finalize;
    };

    class ViteOutputFormat : public TraceOutputFormat
    {
    public:
      inline ViteOutputFunctions& functions() { return m_funcs; }
      
      void open(const std::string& fname) override final;

      void start_trace() override final;
      void declare_threads(int n, double start, double end) override final;
      void declare_state(const std::string& idname, const std::string& fullname, const RGBColord& col) override final;
      void set_state(int t, const std::string& idname, double timepoint, double duration, double max_duration ) override final;
      void finalize_trace(int n, double time) override final;
    private:
      ViteOutputFunctions m_funcs;
    };

  }

}

