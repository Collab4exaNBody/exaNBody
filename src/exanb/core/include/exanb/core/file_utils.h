#pragma once

#include <string>
#include <vector>

namespace exanb
{
  std::string dirname(const std::string& file_name);
  bool is_relative_path(const std::string& path);
  std::string concat_dir_path( const std::string& dirpath, const std::string& filepath  );
  bool resolve_file_path(const std::vector<std::string>& dir_prefixes, std::string& filepath);
  std::string config_file_path(const std::string& base_dir, const std::string& filepath);
  std::string data_file_path( const std::string& filepath );
  void set_install_config_dir(const std::string& cdir);
}
