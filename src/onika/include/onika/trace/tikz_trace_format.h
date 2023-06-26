#pragma once

#include <onika/trace/trace_output_format.h>
#include <onika/color_scale.h>
#include <unordered_map>

namespace onika
{
  namespace trace
  {


    class TikzTraceFormat : public TraceOutputFormat
    {
      static inline constexpr double RES_SCALE = 1.0;
    
    public:
      void open(const std::string& fname) override final;
      void close() override final; 
      void start_trace() override final;
      void declare_threads(int n, double start, double end) override final;
      void declare_state(const std::string& idname, const std::string& fullname, const RGBColord& col) override final;
      void set_state(int t, const std::string& idname, double timepoint, double duration, double max_duration ) override final;
      void finalize_trace(int n, double time) override final;

      //void add_idle_plot( const std::vector<double>& values , double start, double end ) override final;

    private:
      inline void set_num_threads(int n) { m_state.assign(n,ThreadState{0.0,0,0}); m_thread_buffer.clear(); m_thread_buffer.resize(n); }
      inline int num_threads() { return m_state.size(); }

      struct ThreadState
      {
        double tp = 0.0;
        unsigned long c=0;
        unsigned long stid=0;
        std::vector<unsigned long> stc;
      };

      std::unordered_map<unsigned long,std::string> m_state_name;
      std::unordered_map<std::string,unsigned long> m_state_id;
      std::unordered_map<unsigned long,RGBColord> m_state_color;
      
      std::vector<ThreadState> m_state;
      std::vector< std::vector< std::pair<unsigned long,double> > > m_thread_buffer;
      double m_hsize = 28.7;
      double m_vsize = 20.0;
      double m_hscale = 1.0;
      double m_vscale = 1.0;
      double m_start = 0.0;
      double m_end = 0.0;
    };

  }

}


