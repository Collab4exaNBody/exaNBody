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
#include "cmdline.h"
#include <yaml-cpp/yaml.h>
#include <exanb/core/yaml_utils.h>


std::string cmdline_option_to_yaml_int( std::string s , std::string value )
{
  //std::cout << "cmdline_option_to_yaml( '"<<s<<"' , '"<<value<<"' )"<<std::endl;
  std::string::size_type p = s.find('-');
  if( p != std::string::npos )
  {
    value = cmdline_option_to_yaml_int( s.substr(p+1) , value );
    s = s.substr(0,p);
  }
  std::string yaml_str = "{ " ;
  yaml_str += s;
  yaml_str += ": ";
  yaml_str += value;
  yaml_str += " }";
  return yaml_str;
}

std::string cmdline_option_to_yaml( std::string s , std::string value )
{
  std::string opt = "{ configuration: ";
  opt += cmdline_option_to_yaml_int( s,value );
  opt += " }";
  return opt;
}

void command_line_options_to_yaml_config(int argc, char*argv[], int start_opt_arg, YAML::Node& input_data)
{
  using std::string;
  // additional arguments are interpreted as YAML strings that are parsed, and merged on top of files read previously
  for(int a = start_opt_arg; a<argc; a++)
  {
    string opt = argv[a];
    if( opt.find("--") == 0 )
    {
      string optval = "true";
      if( (a+1) < argc )
      {
        if( string(argv[a+1]).find("--") != 0 )
        {
          optval = argv[a+1];
          ++a;
        }
      }
      opt = cmdline_option_to_yaml( opt.substr(2) , optval );
//#     ifndef NDEBUG
//      std::cout << "addon '"<<opt<<"'\n";
//#     endif      
      YAML::Node addon_config = YAML::Load( opt );
      input_data = exanb::merge_nodes( input_data , addon_config );
    }
    /*else
    {
      std::cout<<"skip cmdline arg "<<opt<<"\n";
    }*/
  }
}

