#include <onika/trace/yaml_trace_format.h>
#include <iomanip>

namespace onika
{
  namespace trace
  {

    void YAMLTraceFormat::open(const std::string& fname)
    {
      this->TraceOutputFormat::open(fname+".yml");
    }

    void YAMLTraceFormat::close()
    {
      stream() << "\n";
      this->TraceOutputFormat::close();
    }

    void YAMLTraceFormat::start_trace()
    {
      stream() << "trace:\n";
    }

    void YAMLTraceFormat::finalize_trace(int n, double time)
    {
    }

    void YAMLTraceFormat::declare_state(const std::string& idname, const std::string& fullname, const RGBColord& col)
    {
      auto rgb8 = to_rgb8(col);
      stream() << "  - declare_state: { short: \""<<idname<<"\" , full: \""<<fullname<<"\" , color: ["<<int(rgb8.r)<<","<<int(rgb8.g)<<","<<int(rgb8.b)<<"] }\n";
    }

    void YAMLTraceFormat::set_state(int t, const std::string& idname, double timepoint, double, double)
    {
      stream() << "  - state: { th: "<<t<<", st: \""<<idname<<"\", ti: \""<<std::hexfloat<<timepoint<<"\" }\n";
    }

    void YAMLTraceFormat::add_idle_plot( const std::vector<double>& values , double start, double end )
    {
      stream() << "idle:\n";
      stream() << "  start: "<<start<<"\n";
      stream() << "  end: "<<end<<"\n";
      stream() << "  values:\n";
      for(auto x:values) { stream() << "    - \""<< std::hexfloat << x<<"\"\n"; }
    }

    void YAMLTraceFormat::declare_threads(int n, double start, double end)
    {
      stream() << "  - declare_threads: { threads: "<<n<<" , start: \""<<std::hexfloat<<start<<"\" , end: \""<<std::hexfloat<<end<<"\" }\n";
    }

  }
}



