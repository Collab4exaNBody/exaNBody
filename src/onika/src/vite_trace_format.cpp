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
#include <numeric>
#include <vector>
#include <iomanip>
#include <unordered_map>
#include <random>

#include <onika/trace/vite_trace_format.h>
#define _XSTAMP_VITE_VERBOSE 1

namespace onika
{
  namespace trace
  {

    // ------------------------ VITE format output functions ----------------------

    const ViteOutputFunctions::ViteHeaderFunction ViteOutputFunctions::g_vite_default_header = [](std::ostream& out)
    {
  out <<R"EOF(
%EventDef PajeDefineContainerType 1
% Alias string 
% ContainerType string 
% Name string 
%EndEventDef 
%EventDef PajeDefineStateType 3
% Alias string 
% ContainerType string 
% Name string 
%EndEventDef 
%EventDef PajeDefineEntityValue 6
% Alias string  
% EntityType string  
% Name string  
% Color color 
%EndEventDef  
%EventDef PajeCreateContainer 7
% Time date  
% Alias string  
% Type string  
% Container string  
% Name string  
%EndEventDef  
%EventDef PajeDestroyContainer 8
% Time date  
% Name string  
% Type string  
%EndEventDef  
%EventDef PajeSetState 10
% Time date  
% Type string  
% Container string  
% Value string  
%EndEventDef 
%EventDef PajeSetVariable 51 
% Time date  
% Type string  
% Container string  
% Value double  
%EndEventDef   
1 CT_Prog   0       'Program'
1 CT_Thread CT_Prog 'Thread'
3 ST_ThreadState CT_Thread 'Thread State'
7 0.0 C_Prog CT_Prog 0 'Program'
)EOF";
    };

    const ViteOutputFunctions::ViteDelcareThreadFunction ViteOutputFunctions::g_vite_default_declare_thread = [](std::ostream& out, int t,int n)
    {
      out <<"7 0.0 C_Thread"<<t<<" CT_Thread C_Prog 'Thread "<<t<<"'"<<std::endl;
    };

    const ViteOutputFunctions::ViteDelcareStateFunction ViteOutputFunctions::g_vite_default_declare_state = [](std::ostream& out, const std::string& idname, const std::string& fullname, const RGBColord& col)
    {
      auto [red,green,blue] = col;
      out << "6 "<<idname<<" ST_ThreadState '"<<fullname<<"' '"<<red<<" "<<green<<" "<<blue<<"'"<<std::endl;
      /*
      for(int i=0;i<256;i++)
      {
        auto [r,g,b] = onika::cielab_colormap(i/255.0);
        out << "6 "<<idname+"_"+std::to_string(i)<<" ST_ThreadState '"<<fullname<<"' '"<<r<<" "<<g<<" "<<b<<"'"<<std::endl;
      }
      */
    };

    const ViteOutputFunctions::ViteSetStateFunction ViteOutputFunctions::g_vite_default_set_state =
    [](std::ostream& out, int t, const std::string& idname, double timepoint,double,double)
    {
      out <<"10 "<<std::setprecision(20)<<timepoint<<" ST_ThreadState C_Thread"<<t<<" "<<idname<<std::endl;
    };

    /*
    const ViteOutputFunctions::ViteSetStateFunction ViteOutputFunctions::g_vite_set_state_color_by_duration =
    [](std::ostream& out, int t, const std::string& idname, double timepoint, double duration, double duration_max )
    {
      int64_t relative_duration = std::clamp( static_cast<int64_t>(duration*255.0/duration_max) , 0l , 255l );
      out <<"10 "<<std::setprecision(20)<<timepoint<<" ST_ThreadState C_Thread"<<t<<" "<<idname<<"_"<<relative_duration<<std::endl;
    };
    */

    const ViteOutputFunctions::ViteEndFunction ViteOutputFunctions::g_vite_default_finalize = [](std::ostream& out, int n, double time)
    {
      for(int i=0;i<n;i++)
      {
        out <<"8 "<<std::setprecision(20)<<time<<" C_Thread"<<i<<" CT_Thread"<<std::endl;
      }
      out <<"8 "<<std::setprecision(20)<<time<<" C_Prog CT_Prog"<<std::endl;
    };

    // ------------------------ virtual interface ----------------------

    void ViteOutputFormat::open(const std::string& fname)
    {
      this->TraceOutputFormat::open( fname + ".vite" );
    }

    void ViteOutputFormat::start_trace()
    {
      m_funcs.header(stream());
    }

    void ViteOutputFormat::declare_threads(int n, double , double )
    {
      for(int t=0;t<n;t++) m_funcs.declare_thread(stream(),t,n);
    }

    void ViteOutputFormat::declare_state(const std::string& idname, const std::string& fullname, const RGBColord& col)
    {
      m_funcs.declare_state(stream(),idname,fullname,col);
    }

    void ViteOutputFormat::set_state(int t, const std::string& idname, double timepoint, double duration, double max_duration )
    {
      m_funcs.set_state(stream(),t,idname,timepoint,duration,max_duration);
    }

    void ViteOutputFormat::finalize_trace(int n, double time)
    {
      m_funcs.finalize(stream(),n,time);
    }

  }
}


