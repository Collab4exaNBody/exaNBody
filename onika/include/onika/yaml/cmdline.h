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

#include <string>
#include <yaml-cpp/yaml.h>

namespace onika
{
  namespace yaml
  {
    std::string cmdline_option_to_yaml_int( std::string s , std::string value );
    std::string cmdline_option_to_yaml( std::string s , std::string value );
    void command_line_options_to_yaml_config(int argc, char*argv[], int start_opt_arg, YAML::Node& node);
  }
}

