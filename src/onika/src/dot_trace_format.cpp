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
#include <onika/trace/dot_trace_format.h>
#include <iomanip>

namespace onika
{
  namespace trace
  {

    void DotTraceFormat::open(const std::string& fname)
    {
      this->TraceOutputFormat::open(fname+".neato-n2");
      stream() << "graph root {\n"
               << "graph [outputorder=nodesfirst,overlap=true,splines=false];\n"
               << "node [label=\"\\N\"];\n";
    }

    void DotTraceFormat::close()
    {
      stream() << "\n}\n";
      this->TraceOutputFormat::close();
    }

    void DotTraceFormat::start_trace()
    {
    }

    void DotTraceFormat::finalize_trace(int n, double time)
    {
      stream() << "c0 [style=invis,shape=point,penwidth=0,pos=\""<<m_start*m_scale<<","<<0.0<<"!\"] ;\n";
      stream() << "c1 [style=invis,shape=point,penwidth=0,pos=\""<<m_end*m_scale<<","<<0.0<<"!\"] ;\n";
      stream() << "c2 [style=invis,shape=point,penwidth=0,pos=\""<<m_start*m_scale<<","<<num_threads()<<"!\"] ;\n";
      stream() << "c3 [style=invis,shape=point,penwidth=0,pos=\""<<m_end*m_scale<<","<<num_threads()<<"!\"] ;\n";
      stream() << "}\n";
    }

    void DotTraceFormat::declare_state(const std::string& idname, const std::string& fullname, const RGBColord& col)
    {
      auto stid = m_state_id.size();
      m_state_id[idname] = stid;
      m_state_color[stid] = col;
      // std::cout << "State #"<<stid<<" '"<<idname<<"' aka '"<<fullname<<"' has color "<<onika::format_color_www(col)<<"\n";
    }

    void DotTraceFormat::set_state(int t, const std::string& idname, double timepoint, double, double)
    {
      timepoint *= m_scale;

      if( timepoint < m_state[t].tp )
      {
        std::cout << "overlapping event on thread #"<<t<<" overlap is "<< m_state[t].tp - timepoint << " long" <<std::endl;
        return;
      }

      unsigned long stid = m_state_id[idname];
      if( stid >= m_state[t].stc.size() ) m_state[t].stc.resize(stid+1,0);

      if( timepoint > m_state[t].tp )
      {
        auto col = m_state_color[m_state[t].stid];
        if( m_state[t].stc[m_state[t].stid] % 2 == 1 ) { col.r*=0.85;  col.g*=0.85; col.b*=0.85; }
        double p = ( m_state[t].tp + timepoint )*0.5;
        double w = timepoint - m_state[t].tp;
        stream() << std::setprecision(20) 
                 << "t"<<t<<"e"<<m_state[t].c << " [fillcolor=\""<<onika::format_color_www(col)<<"\""
                 << ",width=\""<<w+(0.01/m_dpi)<<"\",height=\""<<m_hscale+(0.333/m_dpi)<<"\",pos=\""<<p*m_dpi<<","<< (t+0.5)*m_hscale*m_dpi <<"\"] ;" << std::endl;
        ++ m_state[t].c;
        ++ m_state[t].stc[stid];
      }
      
      if( timepoint >= m_state[t].tp )
      {
        m_state[t].tp = timepoint;
        m_state[t].stid = stid;
      }

    }


    void DotTraceFormat::add_idle_plot( const std::vector<double>& values , double start, double end)
    {
      //std::cout << "idle range=["<<start<<";"<<end<<"]"<<std::endl;
      stream() << "subgraph idle {\n"
               << "node [margin=0,label=\"\",shape=\"point\",style=invis];\n"
               << "edge [color=\"#7700FF\",penwidth=\""<<4*RES_SCALE<<"\",constraint=false];\n"
               << "graph [margin=0,dpi=\""<<m_dpi<<"\",bb=\""<<m_start*m_scale<<",0,"<<m_end*m_scale<<","<<num_threads()*m_hscale<<"\",outputorder=nodesfirst,overlap=true,splines=false];\n";

      static constexpr size_t bufsize = 73;
      static constexpr size_t backtrace = 3;
      static_assert( backtrace < bufsize );
      size_t n = values.size();
      std::vector< std::pair<double,double> > buffer;
      size_t counter = 0;

      auto plot_coord = [&](size_t i) -> std::pair<double,double>
        {
          double x = ( start+(i*(end-start)/(n-1)) ) * m_scale * m_dpi;
          double y = values[i] * num_threads() * m_hscale * m_dpi;
          return { x , y };
        };

      auto add_point = [&]( const std::pair<double,double>& p )
        {
          auto [x,y] = p;
          stream() << std::setprecision(20) << "idl"<< counter++ <<" [pos=\""<<x<<","<<y<<"\"] ;\n"; //,style=invis  ,penwidth=0.01
        };

      auto flush_buffer = [&]() -> void
        {
          while( buffer.size() < bufsize ) buffer.push_back( buffer.back() );
          add_point( buffer.front() );
          add_point( buffer.back() );
          stream() << "idl"<<counter-2<< " -- idl"<<counter-1<<" [pos=\"";
          for(size_t j=0;j<bufsize;j++) stream() << ((j>0)?" ":"") << buffer[j].first << "," << buffer[j].second ;
          stream() << "\"] ;\n";
          for(size_t i=0;i<backtrace;i++) buffer[i] = buffer[bufsize-backtrace+i];
          buffer.resize(backtrace);
        };

      for(size_t i=0;i<n;i++)
      {
        buffer.push_back( plot_coord(i) );
        if( buffer.size() == bufsize ) { flush_buffer(); }
      }
      if( ! buffer.empty() ) { flush_buffer(); }
      
      for(int i=0;i<=10;i++)
      {
        double x = (end+(end-start)*0.02) * m_scale * m_dpi;
        double y = i * 0.1 * num_threads() * m_hscale * m_dpi;
        stream() << "idl"<< counter++ <<" [fixedsize=\"false\",style=\"filled\",shape=\"box\",penwidth=0,fillcolor=\"#FFFFFF00\",color=\"#000000\""
                 << ",label=<<FONT POINT-SIZE=\"48pt\" COLOR=\"#7700FF\"><B>-</B> "<< std::setprecision(2) <<i*0.1 <<"</FONT>>"
                 << ",pos=\""<< std::setprecision(20)<<x<<","<<y<<"!\"] ;\n"; //,style=invis  ,penwidth=0.01
      }
      
      stream() << "\n}\n";

    }

    void DotTraceFormat::declare_threads(int n, double start, double end)
    {
      set_num_threads(n);
      m_scale = 19.2*RES_SCALE / (end-start);
      m_hscale = 10.8*RES_SCALE / double(n);
      m_dpi = 72.0;
      m_start = start;
      m_end = end;           
      
      double thread_labels_size = (m_end-m_start)*0.03;
      
      //std::cout << "trace range=["<<m_start<<";"<<m_end<<"]"<<std::endl;
      stream() << "subgraph trace {\n"
               << "node [margin=0,color=\"#FF0000\",fillcolor=\"\",fixedsize=\"true\",height=\"\",label=\"\",penwidth=\"0\",pos=\"\",shape=\"box\",style=\"filled\",width=\"\"];\n"
               << std::setprecision(20) << "graph [margin=0,dpi=\""<<m_dpi<<"\""
	       <<",bb=\""<< (m_start-thread_labels_size) *m_scale*m_dpi <<",0,"<<m_end*m_scale*m_dpi<<","<<n*m_hscale*m_dpi<<"\",outputorder=nodesfirst,overlap=true,splines=false];"<<std::endl;

      for(int i=0;i<n;i++)
      {
        // fontsize="20pt"
        stream() << "tlab"<<i<<" [penwidth=1,style=filled,shape=box,fixedsize=true,fillcolor=\"#FFFFFF\",color=\"#000000\",label=\"Th. "<< std::setfill('0') << std::setw(2) <<i<<"\""
                 <<",pos=\""<<(m_start-thread_labels_size/2)*m_scale*m_dpi<<","<<(i+0.5)*m_hscale*m_dpi<<"\""
                 <<",width=\""<<thread_labels_size*m_scale<<"\", height=\""<<m_hscale<<"\"];\n" ;
      }

    }

  }
}


