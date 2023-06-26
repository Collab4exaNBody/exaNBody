#include <onika/oarray.h>
#include <onika/dac/stencil.h>
#include <unordered_set>
#include <cassert>
#include <cmath>

#include <onika/dag/dot.h>
#include <onika/color_scale.h>

namespace onika
{

  namespace dac
  {

    std::ostream& stencil_dot( std::ostream& out , const AbstractStencil & stencil , const std::function<std::string(uint64_t)> & mask_to_text )
    {
      const size_t nb_elements = stencil.nb_cells();
      const int nd = stencil.m_ndims;
      const auto box_size = stencil.box_size();

      out << "digraph G {\noverlap=\"true\"\nsplines=\"true\"\n";

      out<<"legend [pos=\"0.0,-1.0!\",penwidth=\"0\",label=<<TABLE BORDER=\"1\" CELLBORDER=\"0\"><TR><TD><FONT POINT-SIZE=\"16\" COLOR=\"#007700\">Read Only</FONT></TD></TR><TR><TD><FONT POINT-SIZE=\"16\" COLOR=\"#770000\">Read/Write</FONT></TD></TR></TABLE>>] ;\n";

      for(size_t i=0;i<nb_elements;i++)
      {
        const uint64_t ro_mask_i = stencil.ro_mask(i);
        const uint64_t rw_mask_i = stencil.rw_mask(i);
        if( ( ro_mask_i | rw_mask_i ) != 0 )
        {
          const auto rp = index_to_coord( i , box_size );
          bool center = true; for(int j=0;j<nd;j++) { center = center && (rp[j]+stencil.m_low[j])==0; std::cout<<int(rp[j])<<"/"<<int(stencil.m_low[j])<<" "; }
          
          int x = rp[0];
          int y = 0; if (nd>=2) y = rp[1];
          std::cout<<x<<","<<y<<" center="<<center<<std::endl;
          //std::cout<<"RO="<<stencil.ro_mask(i)<<", RW="<<stencil.rw_mask(i)<<std::endl;
          const char* fc = "#DCDCDC";
          if( center ) fc = "lightblue";
          std::string ro = mask_to_text( stencil.ro_mask(i) );
          std::string rw = mask_to_text( stencil.rw_mask(i) );
          out << "  c" << x << "_" << y << " [label=<<b><font color='#007700'>"<<ro<<"</font> <font color='#770000'>"<<rw<<"</font></b>>"
              <<",fillcolor=\""<<fc<<"\",shape=box,style=\"rounded,filled\",pos=\""<<x<<","<<y<<"!\"] ;\n"; //
        }
      }
      out << "}\n";
      return out;
    }

    template<size_t Nd>
    std::ostream& stencil_dep_dot( std::ostream& out , const std::unordered_set< oarray_t<int,Nd> > & dep_rpos )
    {
      const double pscale=1.5;
      int min_x = 0;
      int min_y = 0;
      for(const auto& d:dep_rpos)
      {
        int x = d[0];
        int y = 0;
        if constexpr (Nd>=2) y = d[1];
        if(x<min_x) min_x=x;
        if(y<min_y) min_y=y;
      }
      
      int md = 0;
      for(const auto& d:dep_rpos)
      {
        md = std::max( std::abs(d[0]) , md );
        if constexpr (Nd>=2) { md = std::max( std::abs(d[1]) , md ); }
      }
      
      out << "digraph G {\noverlap=\"true\"\nsplines=\"true\"\n" ;
      out << "  X [label=\"(0,0)\",fixedsize=\"true\",width=\"0.75\",height=\"0.75\",shape=box,style=\"rounded,filled\",color=black,fillcolor=\""<< onika::format_color_www(onika::cielab_discreet_colormap(0,md)) <<"\",pos=\""<<(-min_x)*pscale<<","<<(-min_y)*pscale<<"!\"] ;\n";
      for(const auto& d:dep_rpos)
      {
        int x = d[0]-min_x;
        int y = 0; if constexpr (Nd>=2) y=d[1]-min_y;
        int dd = 0; for(auto dc:d) dd = std::max(dd,std::abs(dc));
        out << "  c" << x << "_" << y << " [label=\""<<format_array(d) <<"\",fixedsize=\"true\",width=\"0.75\",height=\"0.75\",shape=box,style=\"rounded,filled\",color=black,fillcolor=\""<<onika::format_color_www(onika::cielab_discreet_colormap(dd,md)) <<"\",pos=\""<<x*pscale<<","<<y*pscale<<"!\"] ;\n";
      }
      for(const auto& d:dep_rpos)
      {
        int x = d[0]-min_x;
        int y = 0; if constexpr (Nd>=2) y = d[1]-min_y;
        out << "  X -> c" << x << "_" << y << ";\n";
      }
      out << "}\n";
      return out;
    }
    
    template std::ostream& stencil_dep_dot( std::ostream& out , const std::unordered_set< oarray_t<int,1> > & dep_rpos );
    template std::ostream& stencil_dep_dot( std::ostream& out , const std::unordered_set< oarray_t<int,2> > & dep_rpos );
    template std::ostream& stencil_dep_dot( std::ostream& out , const std::unordered_set< oarray_t<int,3> > & dep_rpos );
    
  }
}

