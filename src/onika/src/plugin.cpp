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

#include <cassert>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <dlfcn.h>

#include <onika/plugin.h>
#include <onika/log.h>
#include <onika/string_utils.h>

#ifndef PLUGIN_DYNLIB_FORMAT
#define PLUGIN_DYNLIB_FORMAT "%s/lib%s.so"
#endif

namespace onika
{
  static std::string g_plugin_to_dynlib_format = PLUGIN_DYNLIB_FORMAT ;
  static std::string g_plugin_default_dir = ONIKA_DEFAULT_PLUGIN_DIR;
  static std::set<std::string> g_loaded_dynlibs;
  static std::string g_loading_plugin;
  static std::string g_plugin_db_filename;
  static PluginDBMap g_plugin_db;
  static bool g_quiet_plugin_register = true;

  void set_quiet_plugin_register(bool b) { g_quiet_plugin_register = b; }
  bool quiet_plugin_register() { return g_quiet_plugin_register; }

  void set_default_plugin_search_dir(const std::string& default_dir)
  {
    g_plugin_default_dir = default_dir;
  }

  void generate_plugin_db( const std::string& filename )
  {
    g_plugin_db_filename = filename;
    std::ofstream fout(g_plugin_db_filename); // zero file
  }

  void plugin_db_register( const std::string& itemCategory, const std::string& itemName )
  {
    if( !g_plugin_db_filename.empty() && !g_loading_plugin.empty() )
    {
      std::ofstream fout(g_plugin_db_filename, std::ios::app);
      fout << g_loading_plugin << " " << itemCategory << " " << itemName << std::endl;
    }
  }

  const PluginDBMap & read_plugin_db( const std::string& filename )
  {
    std::ifstream fin(filename);
    while( fin )
    {
      std::string p, c, i;
      fin >> p >> c >> i;
      if( !p.empty() && !c.empty() && !i.empty() )
      {
        g_plugin_db[c][i] = p;
      }
    }
    return g_plugin_db;
  }
  
  const std::string& suggest_plugin_for( const std::string& itemCategory, const std::string& itemName )
  {
    return g_plugin_db[itemCategory][itemName];
  }


  const std::set<std::string>& loaded_plugins()
  {
    return g_loaded_dynlibs;
  }

  static bool load_plugin_priv( const std::string& filePath)
  {
  	void* handle = dlopen(filePath.c_str(), RTLD_NOW|RTLD_GLOBAL);
  	if( handle == nullptr )
  	{
	    lerr << "plugin '"<<filePath<<"' not loaded ! " << dlerror() << std::endl << std::flush ;
	  }
	  else
	  {
	    g_loaded_dynlibs.insert( filePath );
	  }
    return handle != nullptr;
  }

  size_t load_plugins( const std::vector<std::string> & plugin_files , bool verbose )
  {
    using std::string;
    using std::endl;
    size_t n_loaded = 0;
            
    std::string loading_plugin_backup = g_loading_plugin;

    for( const string& p : plugin_files )
    {
      g_loading_plugin = p;
    
      std::string fp = format_string( g_plugin_to_dynlib_format , g_plugin_default_dir , p );
      
#     ifndef NDEBUG
      const std::string& pname = fp;
#     else
      const std::string& pname = p;
#     endif
      lout<<"+ "<<pname<<endl;
      //if(verbose) { lout<<endl; }
      //else { lout<<"                                                                                           "<< exanb::cret <<std::flush; }

      if( ! load_plugin_priv( fp ) )
      {
        lerr<<"Warning, could not load plugin "<<p<<endl;
      }
      else
      {
        ++ n_loaded;
      }
    }
    //if(!verbose) { lout<<"                                                                                     "<<endl; }

    g_loading_plugin = loading_plugin_backup;
    
    return n_loaded;
  }


}

