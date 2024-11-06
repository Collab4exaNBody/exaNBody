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
#pragma once

#include <iostream>
#include <algorithm>
#include <vector>
#include <string>

#include <onika/stream_utils.h>
#include <onika/oarray.h>

namespace onika
{
  namespace dag
  {
  
    struct ColorMapLegend
    {
      int N=16;
      bool fdp=false;  
    };

    struct DotAttributes
    {
      std::vector<std::string> m_attributes;

      template<class... T>
      inline OStringStream add( const T& ... args )
      {
        m_attributes.emplace_back();
        return OStringStream( m_attributes.back() , args... );
      }
    };

    template<size_t Nd>
    static inline auto prev_node_name( const oarray_t<size_t,Nd>& c )
    {
      return make_streamable_functor( [c](std::ostream& out) -> std::ostream&
        {
          out << "f_" << format_array(c,'_','c','\0');
          return out;
        });
    }

    template<size_t Nd>
    static inline auto node_name( const oarray_t<size_t,Nd>& c , int gs = 1, const oarray_t<size_t,Nd>& g = ZeroArray<size_t,Nd>::zero )
    {
      return make_streamable_functor( [c,gs,g](std::ostream& out) -> std::ostream&
        {
          oarray_t<size_t,Nd> cs;
          for(size_t i=0;i<cs.size();i++) cs[i] = c[i]*gs+g[i];
          out << format_array(cs,'_','c','\0');
          return out;
        });
    }


    inline auto bb_corners(const double bb[4], bool visible=false, const std::string& pfx="")
    {
      return make_streamable_functor( [bb,visible,pfx](std::ostream& out) -> std::ostream&
      {
        if(visible)
        {
          out << pfx << "llcorner [label=\"\",fixedsize=\"true\",width=\"0.05\",height=\"0.05\",style=\"bold\",pos=\""<<bb[0] <<","<<bb[1] <<"!\"];\n";
          out << pfx << "lrcorner [label=\"\",fixedsize=\"true\",width=\"0.05\",height=\"0.05\",style=\"bold\",pos=\""<<bb[2] <<","<<bb[1] <<"!\"];\n";
          out << pfx << "ulcorner [label=\"\",fixedsize=\"true\",width=\"0.05\",height=\"0.05\",style=\"bold\",pos=\""<<bb[0] <<","<<bb[3] <<"!\"];\n";
          out << pfx << "urcorner [label=\"\",fixedsize=\"true\",width=\"0.05\",height=\"0.05\",style=\"bold\",pos=\""<<bb[2] <<","<<bb[3] <<"!\"];\n";
        }
        else
        {
          out << pfx << "llcorner [label=\"\",fixedsize=\"true\",width=\"0.01\",height=\"0.01\",style=invis,pos=\""<<bb[0] <<","<<bb[1] <<"!\"];\n";
          out << pfx << "lrcorner [label=\"\",fixedsize=\"true\",width=\"0.01\",height=\"0.01\",style=invis,pos=\""<<bb[2] <<","<<bb[1] <<"!\"];\n";
          out << pfx << "ulcorner [label=\"\",fixedsize=\"true\",width=\"0.01\",height=\"0.01\",style=invis,pos=\""<<bb[0] <<","<<bb[3] <<"!\"];\n";
          out << pfx << "urcorner [label=\"\",fixedsize=\"true\",width=\"0.01\",height=\"0.01\",style=invis,pos=\""<<bb[2] <<","<<bb[3] <<"!\"];\n";
        }
        return out;
      });
    }

    template<size_t Nd>
    static inline auto node_position( double x, double y, const oarray_t<size_t,Nd>& gc, double gscaling , double gmargin, double gsizex, double gsizey=-1.0 )
    {
      if(gsizey<=0.0) gsizey=gsizex;
      return make_streamable_functor( [x,y,gc,gscaling,gmargin,gsizex,gsizey](std::ostream& out) -> std::ostream&
        {
          double gx = gc[0];
          double gy = 0.0; if constexpr (Nd>=2) gy = gc[1];
          out <<"fixedsize=\"true\",width=\""<<gsizex<<"\",height=\""<<gsizey<<"\",pos=\""<<(x+gx*gscaling+gmargin)<<","<<(y+gy*gscaling+gmargin)<<"!\"" ;
          return out;
        });
    }


    template<size_t Nd>
    static inline auto cluster_name( const oarray_t<size_t,Nd>& c , unsigned int level = 0 )
    {
      return make_streamable_functor( [c,level](std::ostream& out) -> std::ostream&
        {
          out << "cluster_" << format_array(c,'_','c','\0');
          if( level > 0 ) out << "_h"<<level;
          return out;
        });
    }

  }
}

std::ostream& operator << (std::ostream& out , const onika::dag::ColorMapLegend& c );
std::ostream& operator << (std::ostream& out , const onika::dag::DotAttributes& c );

