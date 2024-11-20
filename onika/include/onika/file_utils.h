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
#include <vector>

namespace onika
{
  std::string dirname(const std::string& file_name);
  bool is_relative_path(const std::string& path);
  std::string concat_dir_path( const std::string& dirpath, const std::string& filepath  );
  bool resolve_file_path(const std::vector<std::string>& dir_prefixes, std::string& filepath);
  std::string config_file_path( const std::string& filepath , const std::string& workdir = "." );
  std::string data_file_path( const std::string& filepath );
  void set_install_config_dir(const std::string& cdir);
  void set_data_file_dirs(const std::string& cdir);
  void set_dir_separator(char sep);
}
