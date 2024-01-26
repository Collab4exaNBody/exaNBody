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

