#include <string>
#include <fstream>
#include <cassert>

#include <yaml-cpp/yaml.h>
#include <exanb/core/log.h>
#include <exanb/core/file_utils.h>
#include <exanb/core/yaml_utils.h>

namespace exanb
{

  const YAML::Node & cnode(const YAML::Node &n)
  {
      return n;
  }

  YAML::Node merge_nodes(YAML::Node a, YAML::Node b, bool append_list)
  {
    // sepcial case where lists are to be concatenated
    if( a.IsSequence() && b.IsSequence() && append_list )
    {
      auto c = YAML::Node(YAML::NodeType::Sequence);
      for(YAML::Node n : a) { c.push_back(n); }
      for(YAML::Node n : b) { c.push_back(n); }
      return c;
    }

    if( !b.IsMap() )
    {
      // If b is not a map, merge result is b, unless b is null
      return b.IsNull() ? a : b;
    }
    if( !a.IsMap() )
    {
      // If a is not a map, merge result is b
      return b;
    }
    if ( b.size() == 0 )
    {
      // If a is a map, and b is an empty map, return a
      return a;
    }
        
    // Create a new map 'c' with the same mappings as a, merged with b
    auto c = YAML::Node(YAML::NodeType::Map);
    for (auto n : a)
    {
      if (n.first.IsScalar())
      {
        const std::string & key = n.first.Scalar();
        auto t = YAML::Node(cnode(b)[key]);
        if (t)
        {
          c[n.first] = merge_nodes(n.second, t, append_list);
          continue;
        }
        // special case where key is prefixed with +, indicating that lists are to be concatenated
        auto t2 = YAML::Node(cnode(b)[ "+" + key ]);
        if (t2)
        {
          c[n.first] = merge_nodes(n.second, t2, true);
          continue;
        }
      }
      c[n.first] = n.second;
    }
    
    // Add the mappings from 'b' not already in 'c'
    for (auto n : b)
    {
      if( !n.first.IsScalar() )
      {
        c[n.first] = n.second;
      }
      else
      {
        std::string key = n.first.as<std::string>();
        if( key.find('+')==0 ) { key = key.substr(1); }
        if( !cnode(c)[key] )
        {
          c[key] = n.second;
        }
      }
    }
    
    return c;
  }

  YAML::Node remove_map_key(YAML::Node a, const std::string& k)
  {
    assert( a.IsMap() );
    YAML::Node b(YAML::NodeType::Map);
    for (auto n : a)
    {
      bool copy = true;
      if( n.first.IsScalar() )
      {
        if( n.first.as<std::string>() == k ) { copy = false; }
      }
      if( copy )
      {
        b[ n.first ] = n.second;
      }
    }
    return b;
  }

  void dump_node_to_file(const std::string& file_name, YAML::Node config)
  {
    ldbg << "write config file to '" << file_name << "'" << std::endl;
    std::ofstream fout(file_name.c_str());
    dump_node_to_stream( fout , config );
  }

  // try to load a YAML file, and abort if any yaml execption is raised, printing exception message
  YAML::Node yaml_load_file_abort_on_except(const std::string& file_name)
  {
    try
    {
      YAML::Node node = YAML::LoadFile( file_name );
      return node;
    }
    catch(const YAML::Exception& e)
    {
      lerr << "Error reading YAML file "<<file_name<<std::endl;
      lerr << e.msg <<std::endl;
      lerr << "at line " << e.mark.line <<", column "<<e.mark.column <<std::endl;
      std::abort();
    }
    return YAML::Node();
  }

  static void prefix_config_file_includes(std::vector<std::string>& files , std::string base_dir, std::string file_name )
  {
    using std::string;
    using std::vector;

    file_name = config_file_path(base_dir,file_name);
    base_dir = dirname(file_name);
    // std::cout << " ==> base_dir=" << base_dir << std::endl;

    // ldbg << "yaml include " << file_name << std::endl;
    YAML::Node node = yaml_load_file_abort_on_except( file_name );

    if( node["includes"] )
    {
      vector<string> include_files = node["includes"].as<vector<string> >();
      for(const string& incfile : include_files)
      {
        if( std::find( files.begin() , files.end() , incfile ) == files.end() )
        {
          prefix_config_file_includes( files, base_dir , incfile );
        }
      }
    }
    
    if( std::find( files.begin() , files.end() , file_name ) == files.end() )
    {
      files.push_back( file_name );
    }
  }


  /*!
   * from a single yaml configuration file name, find its complete path
   * and the complete list of files recursively included, only once per file, in the correct include order.
   */
  std::vector<std::string> resolve_config_file_includes(const std::string& app_path, const std::vector<std::string>& file_names )
  {
    using std::string;
    using std::vector;

    // the local file exanb.msp is usually loaded after config_exanb.msp, but we need to read it first in case
    // it defines an alternative config directory via the "config_dir key"
    string app_dir = dirname(app_path);
    string local_default_include_file = app_dir + "/" + XNB_LOCAL_CONFIG_FILE;
    // std::cout << "local_default_include_file = "<<local_default_include_file<<std::endl;
    bool has_local_config_file = false;
    if( std::ifstream(local_default_include_file).good() )
    {
      YAML::Node node = yaml_load_file_abort_on_except( local_default_include_file );
      if( node["configuration"] ) if( node["configuration"]["config_dir"] )
      {
        // std::cout<<"overload config dir with '"<<node["configuration"]["config_dir"].as<string>()<<"'"<<std::endl;
        set_install_config_dir( node["configuration"]["config_dir"].as<string>() );
      }
      has_local_config_file = true;
    }

    // find path to the main base config file 'config_exanb.msp'
    string default_include_file = config_file_path(".",XNB_DEFAULT_CONFIG_FILE);
    // ldbg << "default_include_file = "<<default_include_file<<std::endl;
        
    vector<string> files;
    prefix_config_file_includes( files , app_dir, default_include_file );
    
    // if a file named exanb.msp is found in current working directory, it is included right after default include file
    if( has_local_config_file )
    {
      // std::cout << "using local config file "<<local_default_include_file << std::endl;
      prefix_config_file_includes( files, app_dir, local_default_include_file );
    }
    
    // then add user input file and all subsequent include files
    for(const auto& file_name:file_names) if( ! file_name.empty() )
    {
      prefix_config_file_includes( files , app_dir, file_name );
    }
    return files;
  }

}

