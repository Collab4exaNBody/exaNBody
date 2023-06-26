#pragma once

#include <string>
#include <vector>
#include <set>
#include <map>
#include <cstdlib>

namespace exanb
{
  void generate_plugin_db( const std::string& filename );
  void plugin_db_register( const std::string& itemCategory, const std::string& itemName );

  using PluginDBMap = std::map< std::string , std::map< std::string,std::string> >;
  const PluginDBMap &  read_plugin_db( const std::string& filename );
  const std::string& suggest_plugin_for( const std::string& itemCategory, const std::string& itemName );

  void set_default_plugin_search_dir(const std::string& default_dir);

  void set_quiet_plugin_register(bool b);
  bool quiet_plugin_register();

  // load a set of pluings, return the number of successfuly loaded
  size_t load_plugins( const std::vector<std::string> & plugin_files, bool verbose=false );

  // list of successfuly loaded plugins
  const std::set<std::string>& loaded_plugins();
}

