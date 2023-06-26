#include <iostream>
#include <onika/dag/dot.h>
#include <onika/color_scale.h>

std::ostream& operator << (std::ostream& out , const onika::dag::ColorMapLegend& c )
{
  int header_span = c.fdp ? 2 : (c.N+1);
  const char* header_br = c.fdp ? "<BR/>" : " ";
  //std::ostringstream out;
  
  out<<"<<TABLE BORDER=\"1\" CELLBORDER=\"0\"><TR><TD COLSPAN=\""<<header_span<<"\"><FONT POINT-SIZE=\"16\">Dependency depth"<<header_br<<"</FONT></TD></TR>";
  if(c.fdp)
  {
    for(int i=0;i<=c.N;i++) out<<"<TR><TD BGCOLOR=\""<< onika::format_color_www( onika::cielab_discreet_colormap(i,c.N) )<<"\"></TD><TD><FONT POINT-SIZE=\"24\">"<< i <<"</FONT></TD></TR>";
  }
  else
  {
    out<<"<TR>";
    for(int i=0;i<=c.N;i++) out<<"<TD><FONT POINT-SIZE=\"20\">"<< i <<"</FONT></TD>";
    out<<"</TR><TR>";
    for(int i=0;i<=c.N;i++) out<<"<TD BGCOLOR=\""<< onika::format_color_www( onika::cielab_discreet_colormap(i,c.N) ) <<"\" HEIGHT=\"20\"></TD>";
    out<<"</TR>";
  }
  out<<"</TABLE>>";

  /*std::string s = out.str();
  _out << "\"";
  for(auto c:s) { if(c=='\"') _out <<"\\\""; else _out << c; }
  _out << "\"";
  return _out;
  */
  
  return out;
}

std::ostream& operator << (std::ostream& out , const onika::dag::DotAttributes& c )
{
  if( ! c.m_attributes.empty() )
  {
    out << " [";
    const char* sep="";
    for(const auto& s:c.m_attributes) { out << sep << s; sep=","; }
    out <<"]";
  }
  return out;
}

