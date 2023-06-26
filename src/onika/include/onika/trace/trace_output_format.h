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

