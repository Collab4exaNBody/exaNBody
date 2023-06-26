#include <onika/trace/tikz_trace_format.h>
#include <iomanip>

namespace onika
{
  namespace trace
  {

    void TikzTraceFormat::open(const std::string& fname)
    {
      this->TraceOutputFormat::open(fname+".tikz");
      stream() << "\\documentclass{minimal}\n"
               << "\\usepackage[a4paper, landscape, margin=0in]{geometry}\n"
               << "\\usepackage{tikz}\n"
               << "\\usetikzlibrary{calc}\n"
               << "\\begin{document}\n";
    }

    void TikzTraceFormat::close()
    {
      stream() << "\\end{document}\n";
      this->TraceOutputFormat::close();
    }

    void TikzTraceFormat::start_trace()
    {
    }

    void TikzTraceFormat::finalize_trace(int , double )
    {
      stream() << "\\begin{tikzpicture}\n";
      for(int t=0;t<num_threads();t++)
      {
        stream() <<"\\def\\thl{"<<t*m_vscale<<"}\n"
                 <<"\\def\\thh{"<<(t+1)*m_vscale<<"}\n"
                 <<"\\def\\tp{0.0}\n"                 
                 <<"\\def\\task #1 #2; { \\fill [color=#2] (\\tp,\\thl) rectangle (#1,\\thh); \\def\\tp{#1} }\n";
        for(auto evt : m_thread_buffer[t])
        {
          auto [ stid , timepoint ] = evt;
          if( timepoint < m_state[t].tp )
          {
            std::cout << "event "<<m_state_name[stid]<<" overlaps " <<m_state_name[m_state[t].stid]<<" on thread #"<<t<<" ( "<<m_state[t].tp<<" > "<<timepoint<<" )" <<std::endl;
            //std::abort();
          }
          if( stid >= m_state[t].stc.size() ) m_state[t].stc.resize(stid+1,0);
          if( timepoint > m_state[t].tp )
          {
            int col = m_state[t].stid * 2 + m_state[t].stc[m_state[t].stid] % 2;        
            stream() << std::setprecision(20) << "\\task "<<timepoint<<" C"<<col<<";\n";
                     // << "\\fill [color=C"<<col<<"] ("<<m_state[t].tp<<",\\thl)" <<" rectangle ("<<timepoint<<",\\thh);\n";
            ++ m_state[t].c;
            ++ m_state[t].stc[stid];
          }
          if( timepoint >= m_state[t].tp )
          {
            m_state[t].tp = timepoint;
            m_state[t].stid = stid;
          }
        }
      }
      stream() << "\\end{tikzpicture} \\\\\n";
    }

    void TikzTraceFormat::declare_state(const std::string& idname, const std::string& fullname, const RGBColord& col)
    {
      auto stid = m_state_id.size();
      m_state_id[idname] = stid;
      m_state_color[stid] = col;
      m_state_name[stid] = fullname;
      stream() << "\\definecolor{C"<<stid*2<<'}' << onika::format_color_tex(col) <<'\n';
      stream() << "\\definecolor{C"<<stid*2+1<<'}' << onika::format_color_tex(col*0.85) <<'\n';
      // std::cout << "State #"<<stid<<" '"<<idname<<"' aka '"<<fullname<<"' has color "<<onika::format_color_www(col)<<"\n";
    }

    void TikzTraceFormat::set_state(int t, const std::string& idname, double timepoint, double, double)
    {
      timepoint = (timepoint - m_start) * m_hscale;
      m_thread_buffer[t].push_back( { m_state_id[idname] , timepoint } );
    }

    void TikzTraceFormat::declare_threads(int n, double start, double end)
    {
      set_num_threads(n);
      m_hscale = m_hsize / (end-start);
      m_vscale = m_vsize / double(n);
      m_start = start;
      m_end = end;           
    }

  }
}


