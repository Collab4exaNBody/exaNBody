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

#include <string>
#include <fstream>
#include <iostream>

#include <onika/file_utils.h>
#include <onika/log.h>

#ifndef ONIKA_DEFAULT_CONFIG_DIR
#define ONIKA_DEFAULT_CONFIG_DIR "."
#endif

#ifndef ONIKA_DEFAULT_DATA_DIRS
#define ONIKA_DEFAULT_DATA_DIRS "./data"
#endif

namespace onika
{

  static char g_dir_separator = '/';
  static std::string g_install_config_dir = ONIKA_DEFAULT_CONFIG_DIR;
  static std::string g_data_file_dirs = ONIKA_DEFAULT_DATA_DIRS;

  void set_dir_separator(char sep)
  {
    g_dir_separator = sep;
  }

  void set_install_config_dir(const std::string& cdir)
  {
    g_install_config_dir = cdir;
  }

  void set_data_file_dirs(const std::string& cdir)
  {
    g_data_file_dirs = cdir;
  }

  bool is_relative_path(const std::string& filepath)
  {
    if( filepath.empty() ) { return false; }
    if( filepath[0] == g_dir_separator ) { return false; }
    return true;
  }

  std::string dirname(const std::string& file_name)
  {
    std::string::size_type ls = file_name.find_last_of( g_dir_separator );
    std::string dname = ".";
    if( ls != std::string::npos )
    {
      dname = file_name.substr(0,ls);
    }
    return dname;
  }

  std::string concat_dir_path( const std::string& dirpath, const std::string& filepath)
  {
    if( dirpath.empty() ) { return filepath; }
    else if( dirpath.back() == g_dir_separator ) { return dirpath + filepath; }
    else { return dirpath + g_dir_separator + filepath; }
  }

  bool resolve_file_path(const std::vector<std::string>& dir_prefixes, std::string& filepath)
  {
    if( ! is_relative_path(filepath) ) { return true; }
    
    if( ! std::ifstream(filepath).good() )
    {
      for(auto base_dir : dir_prefixes)
      {
        std::string altpath = concat_dir_path( base_dir , filepath );
        if( std::ifstream(altpath).good() )
        {
          filepath = altpath;
          return true;
        }
        else
        {
          ldbg << "failed path " << altpath << std::endl;
        }
      }
      return false;
    }
    else
    {
      return true;
    }
  }

  std::string config_file_path( const std::string& filepath , const std::string& workdir )
  {
    std::vector<std::string> dirs = { workdir , g_install_config_dir };
    // for( auto d : dirs ) { lout << "config dir "<<d<<std::endl; }
    std::string resolved_path = filepath;
    bool found = resolve_file_path( dirs , resolved_path );
    if( ! found )
    {
      fatal_error() << "configuration file '"<<filepath<<"' not found"<<std::endl;
    }
    return resolved_path;
  }

  std::string data_file_path( const std::string& filepath )
  {
    std::vector<std::string> dirs;
    std::string tmp = g_data_file_dirs;
    std::string::size_type pos = tmp.find(':');
    while(pos != std::string::npos)
    {
      std::string d = tmp.substr(0,pos);
      if( ! d.empty() ) dirs.push_back( d );
      tmp = tmp.substr(pos+1);
      pos = tmp.find(':');
    }
    if(!tmp.empty()) dirs.push_back(tmp);
    
    for( auto d : dirs ) { ldbg << "data dir "<<d<<std::endl; }
    std::string resolved_path = filepath;
    bool found = resolve_file_path( dirs , resolved_path );
    if( ! found )
    {
      ldbg << "Warning: file "<<filepath<<"not found"<<std::endl;
    }
    return resolved_path;
  }

}
