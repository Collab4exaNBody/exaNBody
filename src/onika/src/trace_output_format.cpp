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


