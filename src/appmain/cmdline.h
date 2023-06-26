#pragma once

#include <string>
#include <yaml-cpp/yaml.h>

std::string cmdline_option_to_yaml_int( std::string s , std::string value );
std::string cmdline_option_to_yaml( std::string s , std::string value );
void command_line_options_to_yaml_config(int argc, char*argv[], int start_opt_arg, YAML::Node& node);

